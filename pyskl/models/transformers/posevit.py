import copy as cp
import torch
import torch.nn as nn
from torch import Tensor
from mmcv.runner import load_checkpoint
from typing import Tuple, Optional

from ...utils import Graph, cache_checkpoint
from ..builder import BACKBONES
from .utils import PositionalEncoding


def _init_weights(module) -> None:
    """Based on Huggingface Bert but use truncated normal instead of normal.
    (Timm used trunc_normal in VisionTransformer)
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.trunc_normal_(module.weight, std=.02)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)
    if isinstance(module, nn.Linear) and module.bias is not None:
        nn.init.zeros_(module.bias)


def get_model_config(size: str):
    assert size in ['tiny', 'small', 'base']
    if size == 'tiny':
        hidden_dim, depth, num_heads = 192, 12, 4
    elif size == 'small':
        hidden_dim, depth, num_heads = 384, 12, 6
    else:
        hidden_dim, depth, num_heads = 768, 12, 12
    return hidden_dim, depth, num_heads


@BACKBONES.register_module()
class PoseViT(nn.Module):
    """Spatial-Temporal Transformer
    """
    def __init__(
            self,
            graph_cfg, # Only used to get num of nodes
            in_channels=3,
            size=None,  # if not None, then the model configuration will be overridden
            hidden_dim=128,
            depth=1,
            num_heads=4,
            mlp_ratio=4,
            norm_first=False,
            activation='gelu',
            dropout=0.1,
            use_cls=True,
            layer_norm_eps=1e-6,
            max_position_embeddings=512,
            data_bn_type='VC',
            num_person=2, # * Only used when data_bn_type == 'MVC'
            attention_type='joint_wise',
    ):
        super().__init__()
        assert attention_type in ['frame_wise', 'joint_wise']
        self.attention_type = attention_type

        # graph = Graph(**graph_cfg)
        # A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        #
        # self.data_bn_type = data_bn_type
        # if data_bn_type == 'MVC':
        #     self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        # elif data_bn_type == 'VC':
        #     self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        # else:
        #     self.data_bn = nn.Identity()

        if size is not None:
            hidden_dim, depth, num_heads = get_model_config(size)

        self.hidden_dim = hidden_dim
        self.embd_layer = nn.Linear(in_channels, hidden_dim)

        # cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim)) if use_cls else None

        # positional encoding layers
        self.enc_pe = PositionalEncoding(hidden_dim, max_position_embeddings)
        self.dropout = nn.Dropout(dropout)

        self.LayerNorm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)

        mlp_ratio = int(mlp_ratio)
        # Transformer Encoder
        enc_layer = nn.TransformerEncoderLayer(
            hidden_dim, num_heads, hidden_dim * mlp_ratio, dropout,
            activation, layer_norm_eps, batch_first=True, norm_first=norm_first)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)

        self.init_weights()  # initialization

    def init_weights(self) -> None:
        # Based on Timm
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(_init_weights)

    def forward(self, x):
        N, M, T, V, C = x.size()
        # x = x.permute(0, 1, 3, 4, 2).contiguous()
        # if self.data_bn_type == 'MVC':
        #     x = self.data_bn(x.view(N, M * V * C, T))
        # else:
        #     x = self.data_bn(x.view(N * M, V * C, T))
        # x = x.view(N, M, V, C, T).permute(0, 1, 4, 3, 2).contiguous().view(N * M, T * V, C)

        if self.attention_type == 'joint_wise':
            x = x.view(N * M, T * V, C)
        else:
            x = x.view(N * M, T, V * C)

        # embed the inputs, orig dim -> hidden dim
        x_embd = self.embd_layer(x)
        # prepend the cls token for source if needed
        if self.cls_token is not None:
            cls_token = self.cls_token.expand(x_embd.size(0), -1, -1)
            x_embd = torch.cat((cls_token, x_embd), dim=1)

        # add positional embeddings and encoder forwarding
        x_input = self.enc_pe(x_embd)

        x_input = self.dropout(self.LayerNorm(x_input))

        # (N * M, T * V + 1, C) with cls token
        hidden_state = self.encoder(x_input)

        if self.cls_token is not None:
            hidden_state = hidden_state[:, :1, :]

        hidden_state = hidden_state.reshape((N, M) + hidden_state.shape[1:])

        return hidden_state