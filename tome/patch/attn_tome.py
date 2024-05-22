from typing import Tuple

import torch

from tome.merge import bipartite_soft_matching, merge_source, merge_wavg
from tome.utils import parse_r

from clip.model import ResidualAttentionBlock,VisionTransformer,Transformer,CLIP

class ToMeBlock(ResidualAttentionBlock):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = ToMeAttention(d_model, n_head)
    def forward(self, x: torch.Tensor):
        # print("tomeB")
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        x_attn,metric = self.attn(self.ln_1(x), attn_size)
        x = x + x_attn
        x = x.permute(1,0,2)
        r = self._tome_info["r"].pop(0)
        # print(r)
        if r > 0:
            # Apply ToMe here
            # print("merge")
            merge, _ = bipartite_soft_matching(
                metric,
                r,
                self._tome_info["class_token"],
                self._tome_info["distill_token"],
            )
            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(
                    merge, x, self._tome_info["source"]
                )
            x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])

        x = x.permute(1,0,2)
        x = x + self.mlp(self.ln_2(x))
        return x


class ToMeAttention(torch.nn.MultiheadAttention):
    def forward(
        self, x: torch.Tensor, size: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # print("tomeA")
        # Note: this is copied from timm.models.vision_transformer.Attention with modifications.
        N, B, C = x.shape
        qkv = torch.nn.functional.linear(x.permute(1,0,2), self.in_proj_weight, self.in_proj_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(
            2, 0, 3, 1, 4
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        scale = self.head_dim**-0.5
        attn = (q * scale) @ k.transpose(-2, -1)

        # Apply proportional attention
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out_proj(x)
         
        # Return k as well here
        return x.permute(1,0,2), k.mean(1)


class ToMeTransformer(Transformer):
    def forward(self, x: torch.Tensor):
        self._tome_info["r"] = parse_r(self.layers, self._tome_info["r"])
        return self.resblocks(x)


class ToMeVisionTransformer(VisionTransformer):
    def forward(self, *args, **kwdargs) -> torch.Tensor:
        self._tome_info["size"] = None
        self._tome_info["source"] = None

        return super().forward(*args, **kwdargs)


def apply_patch_vit(model: VisionTransformer, reduction ,trace_source: bool = False, prop_attn: bool = True):
    model.__class__ = ToMeVisionTransformer
    model.r = reduction
    model._tome_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": True,
        "distill_token": False,
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, ResidualAttentionBlock):
            module.__class__ = ToMeBlock
            module._tome_info = model._tome_info
        elif isinstance(module, torch.nn.MultiheadAttention):
            module.__class__ = ToMeAttention
        elif isinstance(module, Transformer):
            module.__class__ = ToMeTransformer
            module._tome_info = model._tome_info


def apply_patch_transformer(model: Transformer, reduction ,trace_source: bool = False, prop_attn: bool = True):
    model.__class__ = ToMeTransformer
    model.r = reduction
    model._tome_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": True,
        "distill_token": False,
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, ResidualAttentionBlock):
            module.__class__ = ToMeBlock
            module._tome_info = model._tome_info
        elif isinstance(module, torch.nn.MultiheadAttention):
            module.__class__ = ToMeAttention