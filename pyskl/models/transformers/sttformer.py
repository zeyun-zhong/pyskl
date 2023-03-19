import torch.nn as nn

from ..builder import BACKBONES
from .utils import STA_Block


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    # nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

def fc_init(fc):
    nn.init.xavier_normal_(fc.weight)
    nn.init.constant_(fc.bias, 0)


@BACKBONES.register_module()
class STTFormer(nn.Module):
    def __init__(self, len_parts, num_joints,
                 num_frames, num_heads, num_channels,
                 kernel_size, use_pes=True, config=None, 
                 att_drop=0, dropout=0, dropout2d=0):
        super().__init__()

        self.len_parts = len_parts
        in_channels = config[0][0]
        self.out_channels = config[-1][1]

        num_frames = num_frames // len_parts
        num_joints = num_joints * len_parts
        
        self.input_map = nn.Sequential(
            nn.Conv2d(num_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1))

        self.blocks = nn.ModuleList()
        for index, (in_channels, out_channels, qkv_dim) in enumerate(config):
            self.blocks.append(STA_Block(in_channels, out_channels, qkv_dim, 
                                         num_frames=num_frames, 
                                         num_joints=num_joints, 
                                         num_heads=num_heads,
                                         kernel_size=kernel_size,
                                         use_pes=use_pes,
                                         att_drop=att_drop))   

        self.drop_out = nn.Dropout(dropout)
        self.drop_out2d = nn.Dropout2d(dropout2d)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
            elif isinstance(m, nn.Linear):
                fc_init(m)

    def forward(self, x):
        N, M, T, V, C = x.size()

        x = x.permute(0, 1, 4, 2, 3).contiguous().view(N * M, C, T, V)
        x = x.view(x.size(0), x.size(1), T // self.len_parts, V * self.len_parts)
        x = self.input_map(x)

        for i, block in enumerate(self.blocks):
            x = block(x)

        # NM, C, T, V
        x = x.view(N, M, -1, self.out_channels)
        x = self.drop_out2d(x)

        return x