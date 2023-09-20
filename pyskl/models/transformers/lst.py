"""Locality-aware Spatiotemporal Transformer
This implementation is based on Pytorch transformer version.
"""

import torch
import torch.nn as nn
from einops import rearrange, pack, unpack
import math

from ..builder import BACKBONES
from .utils import PositionalEncoding
from ..gcns import unit_tcn


def sliding_window_attention_mask(
        seq_len, window_size, neighborhood_size, dtype, with_cls=True):
    """
    Generate sliding window attention mask with a quadratic window shape.

    Args:
    - seq_len: The total sequence length
    - window_size: Size of the attention window
    - neighborhood_size: Number of neighboring windows around the central window

    Returns:
    - mask: A seq_len x seq_len mask where 0 indicates positions that can be attended to, and -1e9 (or a very large negative value) indicates positions that should not be attended to.
    """
    def fn(seq_len, window_size, neighborhood_size):
        mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min)

        num_chunks = math.ceil(seq_len / window_size)
        q_chunks = torch.arange(seq_len).chunk(num_chunks)
        for i, q_chunk in enumerate(q_chunks):
            q_start, q_end = q_chunk[0], q_chunk[-1] + 1
            k_start = max(0, (i - neighborhood_size) * window_size)
            k_end = min(seq_len, (i + neighborhood_size + 1) * window_size)
            mask[q_start: q_end, k_start: k_end] = 0

        return mask

    if not with_cls:
        return fn(seq_len, window_size, neighborhood_size)

    orig_len = seq_len
    seq_len = seq_len - 1
    # mask without considering cls token
    mask = fn(seq_len, window_size, neighborhood_size)
    mask = torch.cat([torch.full((seq_len, 1), torch.finfo(dtype).min), mask], dim=1)
    mask = torch.cat([torch.zeros(1, orig_len, dtype=dtype), mask], dim=0)
    return mask


class TemporalPooling(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride, pooling=False,
                 with_cls=True):
        super().__init__()
        self.pooling = pooling
        self.with_cls = with_cls
        if dim_in == dim_out:
            self.temporal_pool = nn.Identity()
        elif pooling:
            if with_cls:
                self.cls_mapping = nn.Linear(dim_in, dim_out)
            self.temporal_pool = unit_tcn(dim_in, dim_out, kernel_size, stride)
        else:
            self.temporal_pool = nn.Linear(dim_in, dim_out)

    def forward(self, x, v=25):
        # x in B, T, C
        if isinstance(self.temporal_pool, nn.Identity):
            return self.temporal_pool(x)

        if self.pooling:
            # Split cls token and map cls token dimension if any
            if self.with_cls:
                cls, x = x[:, :1, :], x[:, 1:, :]
                cls = self.cls_mapping(cls)

            # TCN
            x = rearrange(x, 'b (t v) c -> b c t v', v=v)
            x = self.temporal_pool(x)
            res = rearrange(x, 'b c t v -> b (t v) c')

            # Concat cls token if any
            if self.with_cls:
                res, pack_shape = pack([cls, res], 'b * c')
        else:
            res = self.temporal_pool(x)
        return res


@BACKBONES.register_module()
class LST(nn.Module):
    """Locality-aware Spatial-Temporal Transformer
    """
    def __init__(
            self,
            in_channels=3,
            hidden_dim=64,
            dim_mul_layers=(4, 7),
            dim_mul_factor=2,
            depth=10,
            num_heads=4,
            mlp_ratio=4,
            norm_first=False,
            activation='relu',
            dropout=0.1,
            use_cls=True,
            layer_norm_eps=1e-6,
            max_joints=25,
            max_frames=100,
            temporal_pooling=True,
            sliding_window=False,
    ):
        super().__init__()

        self.embd_layer = nn.Linear(in_channels, hidden_dim)
        self.norm_first = norm_first
        self.sliding_window = sliding_window
        self.use_cls = use_cls

        # cls token
        self.cls_token = (nn.Parameter(torch.zeros(1, 1, hidden_dim))
                          if use_cls else None)
        self.pos_embed_cls = (nn.Parameter(torch.zeros(1, 1, hidden_dim))
                              if use_cls else None)

        # We use two embeddings, one for joints and one for frames
        self.joint_pe = PositionalEncoding(hidden_dim, max_joints)
        self.frame_pe = PositionalEncoding(hidden_dim, max_frames)

        # Variable hidden dim
        hidden_dims = []
        dim_in = hidden_dim
        for i in range(depth):
            if dim_mul_layers is not None and i in dim_mul_layers:
                dim_out = dim_in * dim_mul_factor
            else:
                dim_out = dim_in
            hidden_dims.append((dim_in, dim_out))
            dim_in = dim_out

        # Transformer Encoder
        self.layers = nn.ModuleList([])
        mlp_ratio = int(mlp_ratio)

        for dim_in, dim_out in hidden_dims:
            self.layers.append(nn.ModuleList([
                # Temporal pool
                TemporalPooling(
                    dim_in, dim_out, 3, dim_mul_factor,
                    pooling=temporal_pooling, with_cls=use_cls),
                # Transformer encoder
                nn.TransformerEncoderLayer(
                    dim_out, num_heads, dim_out * mlp_ratio, dropout,
                    activation, layer_norm_eps, batch_first=True,
                    norm_first=norm_first)
            ]))

        # Variable locality-aware mask
        if sliding_window:
            nb_size = 0
            self.neighborhood_sizes = [nb_size]
            for dim_in, dim_out in hidden_dims[1:]:
                nb_size = nb_size + (dim_in == dim_out)
                self.neighborhood_sizes.append(nb_size)

        self.norm = (nn.LayerNorm(hidden_dims[-1][-1], eps=layer_norm_eps)
                     if norm_first else None)

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        N, M, T, V, C = x.size()
        x = rearrange(x, 'n m t v c -> (n m) t v c')

        # embed the inputs, orig dim -> hidden dim
        x_embd = self.embd_layer(x)

        # add positional embeddings
        x_input = self.joint_pe(x_embd)  # joint-wise
        x_input = self.frame_pe(rearrange(x_input, 'b t v c -> b v t c'))  # frame wise

        # convert to required dim order
        x_input = rearrange(x_input, 'b v t c -> b (t v) c')
        # prepend the cls token for source if needed
        if self.cls_token is not None:
            cls_token = self.cls_token.expand(x_input.size(0), -1, -1)
            cls_token = cls_token + self.pos_embed_cls
            x_input = torch.cat((cls_token, x_input), dim=1)

        hidden_state = x_input
        attn_mask = None
        for i, (temporal_pool, encoder) in enumerate(self.layers):
            # Temporal pooling
            hidden_state = temporal_pool(hidden_state, v=V)

            # Construct attention mask if required
            if self.sliding_window:
                attn_mask = sliding_window_attention_mask(
                    seq_len=hidden_state.size(1),
                    window_size=V,
                    neighborhood_size=self.neighborhood_sizes[i],
                    dtype=hidden_state.dtype,
                    with_cls=self.use_cls,
                ).to(hidden_state.device)
            hidden_state = encoder(hidden_state, attn_mask)

        if self.norm is not None:
            hidden_state = self.norm(hidden_state)

        if self.cls_token is not None:
            hidden_state = hidden_state[:, :1, :]

        hidden_state = rearrange(
            hidden_state, '(n m) tv c -> n m tv c', n=N)

        return hidden_state