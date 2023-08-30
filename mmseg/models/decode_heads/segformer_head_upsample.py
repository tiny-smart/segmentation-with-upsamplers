# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from ..utils import resize
from ..upsamplers import build_upsampler


@MODELS.register_module()
class SegformerHead_Upsample(BaseDecodeHead):
    """The all mlp Head of segformer.

    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self,
                 upsample_cfg=dict(
                     mode='bilinear',
                     guided=False,
                     align_corners=False
                 ),
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

        self.upsample_stages = 6
        self.upsample_cfg = upsample_cfg.copy()
        self.guided_upsample = self.upsample_cfg['guided']
        self.upsample_modules = nn.ModuleList()
        for i in range(self.upsample_stages):
            self.upsample_modules.append(build_upsampler(self.upsample_cfg, in_channels=self.channels, scale_factor=2))

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        x0, x1, x2, x3 = inputs
        x0 = self.convs[0](x0)
        x1 = self.convs[1](x1)
        x2 = self.convs[2](x2)
        x3 = self.convs[3](x3)

        if self.upsample_mode == 'bilinear':
            x3 = resize(x3, size=x0.size()[2:], mode='bilinear', align_corners=False)
            x2 = resize(x2, size=x0.size()[2:], mode='bilinear', align_corners=False)
            x1 = resize(x1, size=x0.size()[2:], mode='bilinear', align_corners=False)
        elif self.is_guided:
            x3 = self.upsamplers[0](x2, x3)
            x3 = self.upsamplers[1](x1, x3)
            x3 = self.upsamplers[2](x0, x3)
            x2 = self.upsamplers[3](x1, x2)
            x2 = self.upsamplers[4](x0, x2)
            x1 = self.upsamplers[5](x0, x1)
        else:
            x3 = self.upsamplers[0](x3)
            x3 = self.upsamplers[1](x3)
            x3 = self.upsamplers[2](x3)
            x2 = self.upsamplers[3](x2)
            x2 = self.upsamplers[4](x2)
            x1 = self.upsamplers[5](x1)

        out = self.fusion_conv(torch.cat([x0, x1, x2, x3], dim=1))

        out = self.cls_seg(out)

        return out
