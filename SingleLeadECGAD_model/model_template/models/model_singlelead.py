import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(3)]
        )
        self.output_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query, key, value = [
            l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linear_layers, (query, key, value))
        ]

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        x = torch.matmul(attn, value)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.output_linear(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ConvEncoder(nn.Module):
    def __init__(self, in_channels, hidden, latent_len=64):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(latent_len)

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(128, hidden, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = x.transpose(1, 2)   # (B, C, T)
        x = self.net(x)         # (B, hidden, T')
        x = self.pool(x)        # (B, hidden, latent_len)
        return x


class ConvDecoder(nn.Module):
    def __init__(self, hidden, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(hidden, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.Conv1d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True),

            nn.Conv1d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(True),

            nn.Conv1d(32, out_channels, kernel_size=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z, target_len):
        x = self.net(z)
        x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
        x = x.transpose(1, 2)   # (B, target_len, out_channels)
        return x


class TemporalClassifierHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 256),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class AttributeHead(nn.Module):
    def __init__(self, in_channels, attr_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 256),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(256, attr_dim),
        )

    def forward(self, x):
        return self.net(x)


class AD_Class(nn.Module):
    def __init__(
        self,
        enc_in,
        hidden=50,
        num_classes=116,
        attr_dim=7,
        latent_len=64,
        attn_heads=2,
    ):
        super().__init__()

        self.channel = enc_in
        self.hidden = hidden

        self.global_encoder = ConvEncoder(enc_in, hidden, latent_len=latent_len)
        self.local_encoder = ConvEncoder(enc_in, hidden, latent_len=latent_len)
        self.trend_encoder = ConvEncoder(enc_in, hidden, latent_len=latent_len)

        self.global_decoder = ConvDecoder(hidden, enc_in + 1)
        self.local_decoder = ConvDecoder(hidden, enc_in + 1)
        self.trend_decoder = ConvDecoder(hidden * 2, enc_in)

        self.attn = MultiHeadedAttention(attn_heads, hidden)
        self.drop = nn.Dropout(0.1)
        self.layer_norm = LayerNorm(hidden)

        self.local_mlp = nn.Sequential(
            nn.Conv1d(hidden * 2, hidden, kernel_size=1, bias=True),
            nn.ReLU(True),
            nn.Conv1d(hidden, hidden, kernel_size=1, bias=True),
        )

        self.global_mlp = nn.Sequential(
            nn.Conv1d(hidden * 2, hidden, kernel_size=1, bias=True),
            nn.ReLU(True),
            nn.Conv1d(hidden, hidden, kernel_size=1, bias=True),
        )

        self.class_head = TemporalClassifierHead(
            in_channels=hidden * 2,
            num_classes=num_classes,
        )

        self.predictor = AttributeHead(
            in_channels=hidden * 2,
            attr_dim=attr_dim,
        )

    def forward(self, global_ecg, local_ecg, trend):
        global_len = global_ecg.shape[1]
        local_len = local_ecg.shape[1]
        trend_len = trend.shape[1]

        latent_global = self.global_encoder(global_ecg)   # (B, h, L)
        latent_local = self.local_encoder(local_ecg)      # (B, h, L)
        latent_trend = self.trend_encoder(trend)          # (B, h, L)

        latent_combine = torch.cat([latent_global, latent_local], dim=1)  # (B, 2h, L)

        latent_combine_t = latent_combine.transpose(1, 2)  # (B, L, 2h)
        compress = latent_combine_t[..., :self.hidden] + latent_combine_t[..., self.hidden:]
        attn_latent = self.attn(compress, compress, compress)
        attn_latent = self.layer_norm(compress + self.drop(attn_latent))
        _ = attn_latent.transpose(1, 2)  # kept for architectural consistency

        latent_local = latent_local + self.local_mlp(latent_combine)
        latent_global = latent_global + self.global_mlp(latent_combine)

        trend_combine = torch.cat([latent_global, latent_trend], dim=1)  # (B, 2h, L)

        gen_global = self.global_decoder(latent_global, target_len=global_len)
        gen_local = self.local_decoder(latent_local, target_len=local_len)
        gen_trend = self.trend_decoder(trend_combine, target_len=trend_len)

        pred_class = self.class_head(trend_combine)
        pred_attribute = self.predictor(trend_combine)

        return (
            gen_global[:, :, 0:self.channel],
            gen_global[:, :, self.channel:self.channel + 1],
        ), (
            gen_local[:, :, 0:self.channel],
            gen_local[:, :, self.channel:self.channel + 1],
        ), gen_trend, pred_attribute, pred_class