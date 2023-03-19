import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from functools import partial, reduce
import operator
import torch.nn.functional as F

from ..builder import BACKBONES
from .utils import round_width, MultiScaleBlock, get_2d_sincos_pos_embed, PatchEmbed


def _get_pool_q_stride(dim_mul_layers, depth, t_scale=2, q_scale=2):
    assert isinstance(dim_mul_layers, (list, tuple))
    pool_q_stride = [[i, 1, 1] for i in range(depth)]
    for layer, _ in dim_mul_layers:
        pool_q_stride[layer][1] = t_scale
        pool_q_stride[layer][2] = q_scale
    return pool_q_stride


@BACKBONES.register_module()
class MST(nn.Module):
    """
    Model builder for Multiscale Skleton Transformer.
    """

    def __init__(
            self,
            in_channels=3,
            embed_dim=64,
            patch_kernel=(3, 1),
            patch_stride=(2, 1),
            patch_padding=(1, 0),
            depth=8,
            num_heads=1,
            spatial_size=25,
            temporal_size=16,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            drop_path=0.2,
            dim_mul_layers=([2, 2.0], [6, 2.0]),
            pool_q_stride_scale=2,
            pool_t_stride_scale=2,
            pool_kvq_kernel=(3, 3),
            pool_kv_stride_adaptive=(1, 4),
            pool_kv_stride_as_q=False,
            norm="layernorm",
            mode="conv",
            use_cls=True,
            pool_first=False,
            rel_pos_spatial_temporal=True,
            rel_pos_zero_init=False,
            residual_pooling=True,
            dim_mul_in_att=True,
            separate_qkv=False,
            use_abs_pos=False,
            use_fixed_sincos_pos=False,
            sep_pos_embed=False,
            spatial_regularization=False,
    ):

        super().__init__()
        # Prepare input.
        self.T, self.V = temporal_size, spatial_size
        self.in_chans = in_channels

        # Prepare backbone
        self.drop_rate = drop_rate
        self.cls_embed_on = use_cls
        head_mul_layers = dim_mul_layers
        pool_q_stride = _get_pool_q_stride(
            dim_mul_layers, depth, t_scale=pool_t_stride_scale, q_scale=pool_q_stride_scale)
        if pool_kv_stride_as_q:
            assert pool_kv_stride_adaptive is None
            pool_kv_stride = pool_q_stride
        else:
            pool_kv_stride = ()

        # Spatial global regularization
        # Only use it when there is no spatial stride
        if spatial_regularization and (pool_q_stride_scale > 1):
            raise ValueError("Only use spatial global regularization "
                             "when there is no spatial stride.")

        # Params for positional embedding
        self.use_abs_pos = use_abs_pos
        self.use_fixed_sincos_pos = use_fixed_sincos_pos
        self.sep_pos_embed = sep_pos_embed
        self.rel_pos_spatial_temporal = rel_pos_spatial_temporal
        if norm == "layernorm":
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        else:
            raise NotImplementedError("Only supports layernorm.")

        self.patch_embed = PatchEmbed(
            dim_in=in_channels,
            dim_out=embed_dim,
            kernel=patch_kernel,
            stride=patch_stride,
            padding=patch_padding,
        )

        input_dims = [temporal_size, spatial_size]
        self.patch_dims = [
            input_dims[i] // patch_stride[i]
            for i in range(len(input_dims))
        ]

        num_patches = reduce(operator.mul, self.patch_dims, 1)  # compatibility for python 3.7

        dpr = [
            x.item() for x in torch.linspace(0, drop_path, depth)
        ]  # stochastic depth decay rule

        if self.cls_embed_on:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            pos_embed_dim = num_patches + 1
        else:
            pos_embed_dim = num_patches

        if self.use_abs_pos:
            if self.sep_pos_embed:
                self.pos_embed_spatial = nn.Parameter(
                    torch.zeros(
                        1, self.patch_dims[1], embed_dim
                    )
                )
                self.pos_embed_temporal = nn.Parameter(
                    torch.zeros(1, self.patch_dims[0], embed_dim)
                )
                if self.cls_embed_on:
                    self.pos_embed_class = nn.Parameter(
                        torch.zeros(1, 1, embed_dim)
                    )
            else:
                self.pos_embed = nn.Parameter(
                    torch.zeros(
                        1,
                        pos_embed_dim,
                        embed_dim,
                    ),
                    requires_grad=not self.use_fixed_sincos_pos,
                )

        if self.drop_rate > 0.0:
            self.pos_drop = nn.Dropout(p=self.drop_rate)

        dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)
        for i in range(len(dim_mul_layers)):
            dim_mul[dim_mul_layers[i][0]] = dim_mul_layers[i][1]
        for i in range(len(head_mul_layers)):
            head_mul[head_mul_layers[i][0]] = head_mul_layers[i][1]

        pool_q = [[] for i in range(depth)]
        pool_kv = [[] for i in range(depth)]
        stride_q = [[] for i in range(depth)]
        stride_kv = [[] for i in range(depth)]

        for i in range(len(pool_q_stride)):
            stride_q[pool_q_stride[i][0]] = pool_q_stride[i][1:]
            if pool_kvq_kernel is not None:
                pool_q[pool_q_stride[i][0]] = pool_kvq_kernel
            else:
                pool_q[pool_q_stride[i][0]] = [
                    s + 1 if s > 1 else s for s in pool_q_stride[i][1:]
                ]

        # If POOL_KV_STRIDE_ADAPTIVE is not None, initialize POOL_KV_STRIDE.
        if pool_kv_stride_adaptive is not None:
            _stride_kv = pool_kv_stride_adaptive
            pool_kv_stride = []
            for i in range(depth):
                if len(stride_q[i]) > 0:
                    _stride_kv = [
                        max(_stride_kv[d] // stride_q[i][d], 1)
                        for d in range(len(_stride_kv))
                    ]
                pool_kv_stride.append([i] + _stride_kv)

        for i in range(len(pool_kv_stride)):
            stride_kv[pool_kv_stride[i][0]] = pool_kv_stride[
                i
            ][1:]
            if pool_kvq_kernel is not None:
                pool_kv[
                    pool_kv_stride[i][0]
                ] = pool_kvq_kernel
            else:
                pool_kv[pool_kv_stride[i][0]] = [
                    s + 1 if s > 1 else s
                    for s in pool_kv_stride[i][1:]
                ]

        self.pool_q = pool_q
        self.pool_kv = pool_kv
        self.stride_q = stride_q
        self.stride_kv = stride_kv

        input_size = self.patch_dims

        self.blocks = nn.ModuleList()

        for i in range(depth):
            num_heads = round_width(num_heads, head_mul[i])
            if dim_mul_in_att:
                dim_out = round_width(
                    embed_dim,
                    dim_mul[i],
                    divisor=round_width(num_heads, head_mul[i]),
                )
            else:
                dim_out = round_width(
                    embed_dim,
                    dim_mul[i + 1],
                    divisor=round_width(num_heads, head_mul[i + 1]),
                )
            attention_block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                input_size=input_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=self.drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                kernel_q=pool_q[i] if len(pool_q) > i else [],
                kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                stride_q=stride_q[i] if len(stride_q) > i else [],
                stride_kv=stride_kv[i] if len(stride_kv) > i else [],
                mode=mode,
                has_cls_embed=self.cls_embed_on,
                pool_first=pool_first,
                rel_pos_spatial_temporal=self.rel_pos_spatial_temporal,
                rel_pos_zero_init=rel_pos_zero_init,
                residual_pooling=residual_pooling,
                dim_mul_in_att=dim_mul_in_att,
                separate_qkv=separate_qkv,
                spatial_regularization=spatial_regularization,
            )

            self.blocks.append(attention_block)
            if len(stride_q[i]) > 0:
                input_size = [
                    size // stride
                    for size, stride in zip(input_size, stride_q[i])
                ]

            embed_dim = dim_out

        self.norm = norm_layer(embed_dim)

        if self.use_abs_pos:
            if self.sep_pos_embed:
                trunc_normal_(self.pos_embed_spatial, std=0.02)
                trunc_normal_(self.pos_embed_temporal, std=0.02)
                if self.cls_embed_on:
                    trunc_normal_(self.pos_embed_class, std=0.02)
            else:
                trunc_normal_(self.pos_embed, std=0.02)
                if self.use_fixed_sincos_pos:
                    pos_embed = get_2d_sincos_pos_embed(
                        self.pos_embed.shape[-1],
                        grid_size=self.patch_dims,
                        cls_token=self.cls_embed_on,
                    )
                    self.pos_embed.data.copy_(
                        torch.from_numpy(pos_embed).float().unsqueeze(0)
                    )

        if self.cls_embed_on:
            trunc_normal_(self.cls_token, std=0.02)

        self.init_weights()

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.02)
            nn.init.constant_(m.weight, 1.0)

    def _get_pos_embed(self, pos_embed, btvc):

        t, v = btvc[1], btvc[2]
        if self.cls_embed_on:
            cls_pos_embed = pos_embed[:, 0:1, :]
            pos_embed = pos_embed[:, 1:]
        txy_num = pos_embed.shape[1]
        p_t, p_v = self.patch_dims
        assert p_t * p_v == txy_num

        if (p_t, p_v) != (t, v):
            new_pos_embed = F.interpolate(
                pos_embed[:, :, :]
                .reshape(1, p_t, p_v, -1)
                .permute(0, 3, 1, 2),
                size=(t, v),
                mode="trilinear",
            )
            pos_embed = new_pos_embed.reshape(1, -1, t * v).permute(0, 2, 1)

        if self.cls_embed_on:
            pos_embed = torch.cat((cls_pos_embed, pos_embed), dim=1)

        return pos_embed

    def forward(self, x, return_attn=False):
        N, M, T, V, C = x.size()
        # Convert to [B, C, T, V]
        x = x.view(N * M, T, V, C).permute(0, 3, 1, 2).contiguous()
        # Convert back to [B, T, V, C']
        x = self.patch_embed(x).permute(0, 2, 3, 1).contiguous()

        btvc = B, T, V, C = x.shape
        x = x.view(B, T * V, C)

        s = 1 if self.cls_embed_on else 0
        if self.use_fixed_sincos_pos:
            x += self.pos_embed[:, s:, :]  # s: on/off cls token

        if self.cls_embed_on:
            cls_tokens = self.cls_token.expand(
                B, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            if self.use_fixed_sincos_pos:
                cls_tokens = cls_tokens + self.pos_embed[:, :s, :]
            x = torch.cat((cls_tokens, x), dim=1)

        if self.use_abs_pos:
            if self.sep_pos_embed:
                pos_embed = self.pos_embed_spatial.repeat(
                    1, self.patch_dims[0], 1
                ) + torch.repeat_interleave(
                    self.pos_embed_temporal,
                    self.patch_dims[1] * self.patch_dims[2],
                    dim=1,
                )
                if self.cls_embed_on:
                    pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)
                x += self._get_pos_embed(pos_embed, btvc)
            elif self.use_fixed_sincos_pos:
                pass  # Since sincos pos embed is already added
            else:
                x += self._get_pos_embed(self.pos_embed, btvc)

        if self.drop_rate:
            x = self.pos_drop(x)

        tv_shape = [T, V]
        for blk in self.blocks:
            x, tv_shape = blk(x, tv_shape)

        x = self.norm(x)
        if self.cls_embed_on:
            x = x[:, :1, :]

        x = x.reshape((N, M) + x.shape[1:])
        return x
