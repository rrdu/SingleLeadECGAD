import argparse
import math
import os
import random
import time
import warnings

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import time_string, convert_secs2time, AverageMeter, generate_trend
from model import AD_Class
from dataloader import DataSet
from losses import AsymmetricLoss


warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Single-lead ECGAD training")

    parser.add_argument("--train_x_path", type=str, required=True)
    parser.add_argument("--train_meta_path", type=str, default=None)
    parser.add_argument("--val_x_path", type=str, default=None)
    parser.add_argument("--val_meta_path", type=str, default=None)
    parser.add_argument("--test_x_path", type=str, default=None)
    parser.add_argument("--test_meta_path", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--dims", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=50)
    parser.add_argument("--latent_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--mask_ratio", type=int, default=30)
    parser.add_argument("--seed", type=int, default=668)

    parser.add_argument("--save_model", type=int, default=1)
    parser.add_argument("--save_path", type=str, default="ckpt/mymodel.pt")

    parser.add_argument("--attr_dim", type=int, default=0)
    parser.add_argument("--num_classes", type=int, default=116)
    parser.add_argument("--num_workers", type=int, default=4)

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def build_loader(x_path, meta_path, args, shuffle, batch_size=None):
    dataset = DataSet(
        x_path=x_path,
        meta_path=meta_path,
        attr_dim=args.attr_dim,
        num_classes=args.num_classes,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size if batch_size is not None else args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers if use_cuda else 0,
        pin_memory=use_cuda,
    )


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    cur_lr = init_lr * 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group["lr"] = cur_lr


def build_global_mask(bs, global_length, mask_ratio, device):
    mask = torch.zeros((bs, global_length, 1), dtype=torch.bool, device=device)

    patch_length = max(1, global_length // 100)
    num_patches = max(1, math.ceil(global_length / patch_length))
    num_mask_patches = min(mask_ratio, num_patches)

    chosen = random.sample(range(num_patches), num_mask_patches)
    for j in chosen:
        start = j * patch_length
        end = min(global_length, (j + 1) * patch_length)
        mask[:, start:end, :] = True

    return mask


def build_local_mask(local_ecg, mask_ratio):
    bs, local_length, dim = local_ecg.shape
    mask_local = local_ecg.clone()

    cut_length = max(1, local_length * mask_ratio // 100)
    cut_length = min(cut_length, local_length)

    if local_length - cut_length - 2 > 1:
        cut_idx = random.randint(1, local_length - cut_length - 2)
    else:
        cut_idx = 0

    mask_local[:, cut_idx:cut_idx + cut_length, :] = 0
    return mask_local


def train_one_epoch(args, model, epoch, train_loader, optimizer):
    model.train()

    total_losses = AverageMeter()
    mse_none = torch.nn.MSELoss(reduction="none")
    cls_loss_fn = AsymmetricLoss(
        gamma_neg=4,
        gamma_pos=10,
        clip=0.05,
        disable_torch_grad_focal_loss=True
    )

    for _, (local_ecg, global_ecg, attribute, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        local_ecg = local_ecg.float().to(device)
        global_ecg = global_ecg.float().to(device)
        attribute = attribute.float().to(device)
        target = target.float().to(device)

        trend = generate_trend(global_ecg)

        bs, local_length, dim = local_ecg.shape
        global_length = global_ecg.shape[1]

        mask_global = global_ecg.clone()
        global_mask = build_global_mask(bs, global_length, args.mask_ratio, device)
        mask_global = torch.mul(mask_global, ~global_mask)

        mask_local = build_local_mask(local_ecg, args.mask_ratio)

        (gen_global, global_var), (gen_local, local_var), gen_trend, gen_attr, prediction = model(
            mask_global, mask_local, trend
        )

        global_err = (gen_global - global_ecg) ** 2
        local_err = (gen_local - local_ecg) ** 2
        trend_err = (gen_trend - global_ecg) ** 2
        attr_err = mse_none(gen_attr, attribute)

        l_global = torch.mean(torch.exp(-global_var) * global_err) + torch.mean(global_var)
        l_local = torch.mean(torch.exp(-local_var) * local_err) + torch.mean(local_var)
        l_trend = torch.mean(trend_err)
        l_attr = torch.mean(attr_err)
        l_class = cls_loss_fn(prediction, target)

        final_loss = l_global + l_local + l_trend + l_attr + l_class

        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()

        total_losses.update(final_loss.item(), bs)

    print(f"Train Epoch: {epoch} Total_Loss: {total_losses.avg:.6f}")


@torch.no_grad()
def evaluate(model, loader, split_name="Eval"):
    model.eval()
    sigmoid = torch.nn.Sigmoid()

    labels = []
    scores = []

    for _, (local_ecg, global_ecg, attribute, target) in tqdm(enumerate(loader), total=len(loader)):
        local_ecg = local_ecg.float().to(device)
        global_ecg = global_ecg.float().to(device)
        target = target.float().to(device)

        trend = generate_trend(global_ecg)
        (_, _), (_, _), _, _, prediction = model(global_ecg, local_ecg, trend)
        prediction = sigmoid(prediction)

        scores.append(prediction.detach().cpu())
        labels.append(target.detach().cpu())

    scores = torch.cat(scores).numpy()
    labels = torch.cat(labels).numpy()

    auc_result = roc_auc_score(labels, scores)
    print(f"{split_name} AUC: {auc_result:.3f}")
    return auc_result


def main():
    args = parse_args()
    set_seed(args.seed)

    print("args:", args)

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    train_loader = build_loader(
        x_path=args.train_x_path,
        meta_path=args.train_meta_path,
        args=args,
        shuffle=True,
        batch_size=args.batch_size,
    )

    val_loader = None
    if args.val_x_path is not None:
        val_loader = build_loader(
            x_path=args.val_x_path,
            meta_path=args.val_meta_path,
            args=args,
            shuffle=False,
            batch_size=1,
        )

    test_loader = None
    if args.test_x_path is not None:
        test_loader = build_loader(
            x_path=args.test_x_path,
            meta_path=args.test_meta_path,
            args=args,
            shuffle=False,
            batch_size=1,
        )

    model = AD_Class(
        enc_in=args.dims,
        hidden=args.hidden,
        num_classes=args.num_classes,
        attr_dim=args.attr_dim,
        latent_len=args.latent_len,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    start_time = time.time()
    epoch_time = AverageMeter()
    best_auc = -float("inf")

    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(optimizer, args.lr, epoch, args)

        need_hour, need_mins, need_secs = convert_secs2time(
            epoch_time.avg * (args.epochs - epoch)
        )
        need_time = "[Need: {:02d}:{:02d}:{:02d}]".format(
            need_hour, need_mins, need_secs
        )
        print(f"{epoch:3d}/{args.epochs:3d} ----- [{time_string()}] {need_time}")

        epoch_time.update(time.time() - start_time)
        start_time = time.time()

        train_one_epoch(args, model, epoch, train_loader, optimizer)

        eval_loader = val_loader if val_loader is not None else test_loader
        eval_name = "Val" if val_loader is not None else "Test"

        if eval_loader is not None:
            auc_result = evaluate(model, eval_loader, split_name=eval_name)

            if auc_result > best_auc:
                best_auc = auc_result
                if args.save_model == 1:
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "best_auc": best_auc,
                            "args": vars(args),
                        },
                        args.save_path,
                    )

    if best_auc > -float("inf"):
        print("Final best AUC:", best_auc)

    if test_loader is not None and val_loader is not None:
        print("Running final test evaluation using current model...")
        evaluate(model, test_loader, split_name="Test")


if __name__ == "__main__":
    main()