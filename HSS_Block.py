from models.HilbertSS2D import HSS
import torch
import torch.nn as nn
from typing import Optional, Callable

from functools import partial

class HSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,  # 96
            # 归一化操作
            # 通过将 norm_layer 作为 Callable，可以在模型定义时灵活传入不同的归一化层。
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),  # nn.LN
            d_state: int = 16,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = HSS(d_model=hidden_dim, d_state=d_state)

    def forward(self, input: torch.Tensor):
        x = input + self.self_attention(self.ln_1(input))
        return x

# 每个HSSLayer包含2个HSS Block和1个downsample
class HSSLayer(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            d_state=16,
            norm_layer=nn.LayerNorm,
            # 下采样操作，此处为 PatchMergin2D
            downsample=None,
            use_checkpoint=False,  # 不适用checkpoint来节省内存
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint
        # for 循环的作用是用来创建一个包含多个 VSSBlock 实例的列表。
        # 这些 VSSBlock 实例会被存储在 self.blocks 这个 nn.ModuleList 中，并在前向传播时被依次调用
        self.blocks = nn.ModuleList([
            HSSBlock(
                hidden_dim=dim,
                norm_layer=norm_layer,
                d_state=d_state,
            )
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        # 遍历HSS Block
        for blk in self.blocks:
            x = blk(x)

        # 遍历完HSS Block后，进行下采样，尺寸减半，维度翻倍
        if self.downsample is not None:
            x = self.downsample(x)

        return x