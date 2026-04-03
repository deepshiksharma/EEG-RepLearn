import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, stride=1, padding=None, num_groups=8):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2

        assert out_ch % num_groups == 0

        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
        self.norm = nn.GroupNorm(num_groups, out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class DownsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = ConvBlock1D(in_ch, out_ch, stride=2)

    def forward(self, x):
        return self.block(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock1D(in_ch, out_ch)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=False)
        x = self.conv(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, depth, nhead, ff_dim, dropout=0.0):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            batch_first=True,
            activation='gelu',
            dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, x):
        return self.encoder(x)


class ConvDecoderWaveform(nn.Module):
    def __init__(self, num_channels, hidden_ch=64, num_groups=8):
        super().__init__()

        assert hidden_ch % num_groups == 0

        self.in_conv = nn.Conv1d(num_channels, hidden_ch, kernel_size=1)
        self.norm_in = nn.GroupNorm(num_groups, hidden_ch)

        self.conv1 = nn.Conv1d(hidden_ch, hidden_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, hidden_ch)

        self.conv2 = nn.Conv1d(hidden_ch, hidden_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, hidden_ch)

        self.out_conv = nn.Conv1d(hidden_ch, num_channels, kernel_size=1)

        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.norm_in(self.in_conv(x)))
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))
        x = self.out_conv(x)
        return x
    

class The_Encoder(nn.Module):
    def __init__(self, num_channels=16, embed_dim=128,
                 transformer_depth=2, nhead=4, ff_dim=512, dropout=0.0):
        super().__init__()

        # downsampling
        self.enc1 = DownsampleBlock(num_channels, 32)   # 2000 -> 1000
        self.enc2 = DownsampleBlock(32, 64)             # 1000 -> 500
        self.enc3 = DownsampleBlock(64, embed_dim)      # 500 -> 250
        # positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, 250, embed_dim))
        # transformer
        self.transformer = TransformerBlock(embed_dim, transformer_depth, nhead, ff_dim, dropout)

    def forward(self, x, apply_transformer=True):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)            # (B, 128, 250)

        x = x.permute(0, 2, 1)      # (B, 250, 128)
        x = x + self.pos_embed

        if apply_transformer:
            x = self.transformer(x)     # (B, 250, 128)     

        return x


class EEG_MaskedAutoencoder(nn.Module):
    def __init__(self, num_channels=16, T=2000, mask_ratio=0.6, embed_dim=128,
                 transformer_depth=2, nhead=4, ff_dim=512, conv_decoder_hidden=64, dropout=0.0):
        super().__init__()

        assert T == 2000

        self.T = T
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        
        # The Encoder
        self.encoder = The_Encoder(
            num_channels=num_channels,
            embed_dim=embed_dim,
            transformer_depth=transformer_depth,
            nhead=nhead,
            ff_dim=ff_dim,
            dropout=dropout
        )

        # mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # The Decoder
        # downsampling
        self.up1 = UpsampleBlock(embed_dim, 96)   # 250 -> 500
        self.up2 = UpsampleBlock(96, 64)          # 500 -> 1000
        self.up3 = UpsampleBlock(64, 32)          # 1000 -> 2000
        # project features to number of signal channels
        self.final_conv = nn.Conv1d(32, num_channels, kernel_size=1)
        # waveform refinement
        self.refine = ConvDecoderWaveform(num_channels, conv_decoder_hidden)
    
    def forward(self, x):
        B, C, T = x.shape
        assert C == self.num_channels and T == self.T

        x_input = x

        # encoder
        x = self.encoder(x, apply_transformer=False)    # (B, 250, 128)

        B, N, D = x.shape
        device = x.device

        # masking
        num_mask = int(N * self.mask_ratio)

        noise = torch.rand(B, N, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)

        masked_idx = ids_shuffle[:, :num_mask]
        keep_idx = ids_shuffle[:, num_mask:]

        mask = torch.ones(B, N, dtype=torch.bool, device=device)
        mask.scatter_(1, keep_idx, False)

        keep_idx_exp = keep_idx.unsqueeze(-1).expand(-1, -1, D)
        visible_tokens = torch.gather(x, 1, keep_idx_exp)

        encoded = self.encoder.transformer(visible_tokens)

        # reconstruct full sequence
        full_tokens = torch.zeros_like(x)

        full_tokens.scatter_(1, keep_idx_exp, encoded)

        mask_tokens = self.mask_token.expand(B, num_mask, D)
        masked_idx_exp = masked_idx.unsqueeze(-1).expand(-1, -1, D)
        full_tokens.scatter_(1, masked_idx_exp, mask_tokens)

        # decode
        x = full_tokens.permute(0, 2, 1)    # (B, 128, 250)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)

        x = self.final_conv(x)
        x = self.refine(x)

        # targets
        target = x_input
        recon = x

        # masked targets
        masked_idx_exp = masked_idx

        decoded_masked = torch.gather(recon, 2, masked_idx_exp.unsqueeze(1).expand(-1, C, -1))
        target_masked = torch.gather(target, 2, masked_idx_exp.unsqueeze(1).expand(-1, C, -1))
        
        return decoded_masked, target_masked, mask, recon
