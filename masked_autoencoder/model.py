import torch
import torch.nn as nn


class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=11, stride=1, padding=None, num_groups=8):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2

        assert out_ch % num_groups == 0, f'out_ch={out_ch} must be divisible by num_groups={num_groups}'
        
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
        self.norm = nn.GroupNorm(num_groups, out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, in_chans, patch_size, embed_dim):
        super().__init__()
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)        # (B, D, N)
        x = x.permute(0, 2, 1)  # (B, N, D)
        return x


class MAE_Encoder(nn.Module):
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


class MAE_Decoder(nn.Module):
    def __init__(self, embed_dim, decoder_dim, depth, nhead, ff_dim,
                 patch_size, num_channels, dropout=0.0):
        super().__init__()

        self.input_proj = nn.Linear(embed_dim, decoder_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            batch_first=True,
            activation='gelu',
            dropout=dropout
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=depth)

        self.output_proj = nn.Linear(decoder_dim, patch_size * num_channels)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.decoder(x)
        x = self.output_proj(x)
        return x


class ConvDecoderWaveform(nn.Module):
    # conv waveform refinement with GroupNorm
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


class EEG_MaskedAutoencoder(nn.Module):
    def __init__(self, num_channels=16, T=2000, patch_size=25, mask_ratio=0.7, embed_dim=128,
                 encoder_depth=6, decoder_depth=4, decoder_dim=64, nhead=6, ff_dim=768,
                 conv_stem_channels=(32, 64, 64), conv_decoder_hidden=64, dropout=0.0):
        super().__init__()

        assert T is not None and T % patch_size == 0

        self.T = T
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio

        # conv stem
        layers = []
        in_ch = num_channels
        for out_ch in conv_stem_channels:
            layers.append(ConvBlock1D(in_ch, out_ch))
            in_ch = out_ch
        self.conv_stem = nn.Sequential(*layers)

        stem_out = conv_stem_channels[-1]

        # patch embedding
        self.patch_embed = PatchEmbed(stem_out, patch_size, embed_dim)
        self.norm_patch = nn.LayerNorm(embed_dim)

        # positional embeddings
        num_patches = T // patch_size
        self.num_patches = num_patches

        self.pos_embed_enc = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        self.pos_embed_dec = nn.Parameter(torch.randn(1, num_patches, embed_dim))

        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # transformer
        self.encoder = MAE_Encoder(embed_dim, encoder_depth, nhead, ff_dim, dropout)
        self.decoder_tok = MAE_Decoder(embed_dim, decoder_dim, decoder_depth,
                                       nhead, ff_dim, patch_size, num_channels, dropout)

        self.conv_decoder = ConvDecoderWaveform(num_channels, conv_decoder_hidden)

    def forward(self, x):
        """
        x: (B, C, T)
        returns:
            decoded_masked (B, num_mask, patch_dim)
            target_masked  (B, num_mask, patch_dim)
            mask           (B, N)
        """
        B, C, T = x.shape
        assert C == self.num_channels and T == self.T

        x_input = x  # preserve raw EEG for target

        # conv stem
        x = self.conv_stem(x)

        # patch tokens
        x = self.patch_embed(x)
        x = self.norm_patch(x)
        x = x + self.pos_embed_enc

        B, N, D = x.shape
        device = x.device

        # random masking
        num_mask = int(N * self.mask_ratio)

        noise = torch.rand(B, N, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)

        masked_idx = ids_shuffle[:, :num_mask]
        keep_idx = ids_shuffle[:, num_mask:]

        mask = torch.ones(B, N, dtype=torch.bool, device=device)
        mask.scatter_(1, keep_idx, False)

        # encode visible tokens
        keep_idx_exp = keep_idx.unsqueeze(-1).expand(-1, -1, D)
        visible_tokens = torch.gather(x, 1, keep_idx_exp)

        encoded = self.encoder(visible_tokens)

        # reconstruct full sequence
        decoder_input = torch.zeros_like(x)

        decoder_input.scatter_(1, keep_idx_exp, encoded)

        mask_tokens = self.mask_token.expand(B, num_mask, D)
        masked_idx_exp = masked_idx.unsqueeze(-1).expand(-1, -1, D)
        decoder_input.scatter_(1, masked_idx_exp, mask_tokens)

        decoder_input = decoder_input + self.pos_embed_dec

        # decode
        token_patches = self.decoder_tok(decoder_input)

        # reconstruct waveform
        tokens = token_patches.view(B, N, self.num_channels, self.patch_size)
        wave = tokens.permute(0, 2, 1, 3).contiguous().view(B, self.num_channels, T)

        wave = self.conv_decoder(wave)

        # targets (raw signal)
        target = x_input.unfold(2, self.patch_size, self.patch_size)
        target = target.permute(0, 2, 1, 3).contiguous().view(B, N, -1)

        recon = wave.unfold(2, self.patch_size, self.patch_size)
        recon = recon.permute(0, 2, 1, 3).contiguous().view(B, N, -1)

        patch_dim = target.shape[-1]

        masked_idx_exp = masked_idx.unsqueeze(-1).expand(-1, -1, patch_dim)

        decoded_masked = torch.gather(recon, 1, masked_idx_exp)
        target_masked = torch.gather(target, 1, masked_idx_exp)

        return decoded_masked, target_masked, mask
