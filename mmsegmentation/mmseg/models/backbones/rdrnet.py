import math
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple, Union, List
from mmcv.cnn import ConvModule, build_norm_layer
from mmengine.model import BaseModule

from mmseg.models.utils import RPPM, BasicBlock, Bottleneck, resize
from mmseg.registry import MODELS
from mmseg.utils import OptConfigType


@MODELS.register_module()
class RDRNet(BaseModule):
    """RDRNet backbone.

    Args:
        in_channels (int): Number of input image channels. Default: 3
        channels: (int): The base channels of RDRNet. Default: 32
        ppm_channels (int): The channels of PPM module. Default: 128
        num_blocks_per_stage (List[int]): The number of blocks with a
            stride of 1 from stage 2 to stage 6. '[4, 3, [5, 4], [5, 4], [1, 1]]'
            corresponding RDRNet-S-Simple, RDRNet-S and RDRNet-M,
            '[6, 5, [7, 6], [7, 6], [2, 2]]' corresponding RDRNet-L.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False
        norm_cfg (dict): Config dict to build norm layer.
            Default: dict(type='BN', requires_grad=True)
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU', inplace=True)
        init_cfg (dict, optional): Initialization config dict.
            Default: None
        deploy (bool): Whether in deploy mode. Default: False
    """

    def __init__(self,
                 in_channels: int = 3,
                 channels: int = 32,
                 ppm_channels: int = 128,
                 num_blocks_per_stage: List[int] = [4, 3, [5, 4], [5, 4], [1, 1]],
                 align_corners: bool = False,
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True, momentum=0.03, eps=0.001),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 init_cfg: OptConfigType = None,
                 deploy: bool = False):
        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.ppm_channels = ppm_channels
        self.num_blocks_per_stage = num_blocks_per_stage

        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners
        self.deploy = deploy

        # stage 1-3
        self.stem = nn.Sequential(
            # stage1
            RB(in_channels=in_channels, out_channels=channels, stride=2, norm_cfg=self.norm_cfg, deploy=self.deploy),

            # stage2
            RB(in_channels=channels, out_channels=channels, stride=2, norm_cfg=self.norm_cfg, deploy=self.deploy),
            *[RB(in_channels=channels, out_channels=channels, stride=1, norm_cfg=self.norm_cfg, deploy=self.deploy) for
              _ in range(self.num_blocks_per_stage[0])],

            # stage3
            RB(in_channels=channels, out_channels=channels * 2, stride=2, norm_cfg=self.norm_cfg, deploy=self.deploy),
            *[RB(in_channels=channels * 2, out_channels=channels * 2, stride=1, norm_cfg=self.norm_cfg,
                 deploy=self.deploy) for _ in range(self.num_blocks_per_stage[1])],
        )
        self.relu = nn.ReLU()

        # semantic branch
        self.semantic_branch_layers = nn.ModuleList()
        self.semantic_branch_layers.append(
            nn.Sequential(
                RB(in_channels=channels * 2, out_channels=channels * 4, stride=2, norm_cfg=self.norm_cfg,
                   deploy=self.deploy),
                *[RB(in_channels=channels * 4, out_channels=channels * 4, stride=1, norm_cfg=self.norm_cfg,
                     deploy=self.deploy) for _ in range(self.num_blocks_per_stage[2][0] - 1)],
                RB(in_channels=channels * 4, out_channels=channels * 4, stride=1, norm_cfg=self.norm_cfg, act=False,
                   deploy=self.deploy),
            )
        )
        self.semantic_branch_layers.append(
            nn.Sequential(
                RB(in_channels=channels * 4, out_channels=channels * 8, stride=2, norm_cfg=self.norm_cfg,
                   deploy=self.deploy),
                *[RB(in_channels=channels * 8, out_channels=channels * 8, stride=1, norm_cfg=self.norm_cfg,
                     deploy=self.deploy) for _ in range(self.num_blocks_per_stage[3][0] - 1)],
                RB(in_channels=channels * 8, out_channels=channels * 8, stride=1, norm_cfg=self.norm_cfg, act=False,
                   deploy=self.deploy),
            )
        )
        self.semantic_branch_layers.append(
            nn.Sequential(
                self._make_layer(
                    block=Bottleneck,
                    inplanes=channels * 8,
                    planes=channels * 8,
                    num_blocks=self.num_blocks_per_stage[4][0],
                    stride=2),
            )
        )

        # bilateral fusion
        self.compression_1 = ConvModule(
            channels * 4,
            channels * 2,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        self.down_1 = ConvModule(
            channels * 2,
            channels * 4,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=None)

        self.compression_2 = ConvModule(
            channels * 8,
            channels * 2,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        self.down_2 = nn.Sequential(
            ConvModule(
                channels * 2,
                channels * 4,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                channels * 4,
                channels * 8,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=None))

        # detail branch
        self.detail_branch_layers = nn.ModuleList()
        self.detail_branch_layers.append(
            nn.Sequential(
                *[RB(in_channels=channels * 2, out_channels=channels * 2, stride=1, norm_cfg=self.norm_cfg,
                     deploy=self.deploy) for _ in range(self.num_blocks_per_stage[2][1] - 1)],
                RB(in_channels=channels * 2, out_channels=channels * 2, stride=1, norm_cfg=self.norm_cfg, act=False,
                   deploy=self.deploy),
            )
        )
        self.detail_branch_layers.append(
            nn.Sequential(
                *[RB(in_channels=channels * 2, out_channels=channels * 2, stride=1, norm_cfg=self.norm_cfg,
                     deploy=self.deploy) for _ in range(self.num_blocks_per_stage[3][1] - 1)],
                RB(in_channels=channels * 2, out_channels=channels * 2, stride=1, norm_cfg=self.norm_cfg, act=False,
                   deploy=self.deploy),
            )
        )
        self.detail_branch_layers.append(
            self._make_layer(
                block=Bottleneck,
                inplanes=channels * 2,
                planes=channels * 2,
                num_blocks=self.num_blocks_per_stage[4][1],
            )
        )

        self.spp = RPPM(
            channels * 16, ppm_channels, channels * 4, num_scales=5, norm_cfg=self.norm_cfg, deploy=self.deploy)

        self.kaiming_init()

    def forward(self, x):
        """Forward function."""
        out_size = (math.ceil(x.shape[-2] / 8), math.ceil(x.shape[-1] / 8))

        # stage 1-3
        x = self.stem(x)

        # stage4
        x_s = self.semantic_branch_layers[0](x)
        x_d = self.detail_branch_layers[0](x)
        comp_c = self.compression_1(self.relu(x_s))
        x_s = x_s + self.down_1(self.relu(x_d))
        x_d = x_d + resize(comp_c,
                           size=out_size,
                           mode='bilinear',
                           align_corners=self.align_corners)
        if self.training:
            temp_context = x_d.clone()

        # stage5
        x_s = self.semantic_branch_layers[1](self.relu(x_s))
        x_d = self.detail_branch_layers[1](self.relu(x_d))
        comp_c = self.compression_2(self.relu(x_s))
        x_s = x_s + self.down_2(self.relu(x_d))
        x_d = x_d + resize(comp_c,
                           size=out_size,
                           mode='bilinear',
                           align_corners=self.align_corners)

        # stage6
        x_d = self.detail_branch_layers[2](self.relu(x_d))
        x_s = self.semantic_branch_layers[2](self.relu(x_s))
        x_s = self.spp(x_s)
        x_s = resize(
            x_s,
            size=out_size,
            mode='bilinear',
            align_corners=self.align_corners)

        return (temp_context, x_d + x_s) if self.training else x_d + x_s

    def _make_layer(self, block, inplanes, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                build_norm_layer(self.norm_cfg, planes * block.expansion)[1])

        layers = [
            block(
                in_channels=inplanes,
                channels=planes,
                stride=stride,
                norm_cfg=self.norm_cfg,
                downsample=downsample)
        ]
        inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(
                block(
                    in_channels=inplanes,
                    channels=planes,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg_out=None if i == num_blocks - 1 else self.act_cfg))

        return nn.Sequential(*layers)

    def switch_to_deploy(self):
        for m in self.modules():
            if isinstance(m, RB):
                m.switch_to_deploy()
        self.spp.switch_to_deploy()
        self.deploy = True

    def kaiming_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class Block1x1(BaseModule):
    """The 1x1 path of the Reparameterizable Block.

        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            stride (int or tuple): Stride of the convolution. Default: 1
            padding (int, tuple): Padding added to all four sides of
                the input. Default: 1
            bias (bool) : Whether to use bias.
                Default: True
            norm_cfg (dict): Config dict to build norm layer.
                Default: dict(type='BN', requires_grad=True)
            deploy (bool): Whether in deploy mode. Default: False
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: Union[int, Tuple[int]] = 1,
                 padding: Union[int, Tuple[int]] = 0,
                 bias: bool = True,
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True, momentum=0.03, eps=0.001),
                 deploy: bool = False):
        super().__init__()

        self.deploy = deploy
        if self.deploy:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=padding, bias=True)
        else:
            self.conv1 = ConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=padding,
                bias=bias,
                norm_cfg=norm_cfg,
                act_cfg=None)
            self.conv2 = ConvModule(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=padding,
                bias=bias,
                norm_cfg=norm_cfg,
                act_cfg=None)

    def forward(self, x):
        if self.deploy:
            x = self.conv(x)
        else:
            x = self.conv1(x)
            x = self.conv2(x)

        return x

    def _fuse_bn_tensor(self, conv: nn.Module):
        kernel = conv.conv.weight
        bias = conv.conv.bias
        running_mean = conv.bn.running_mean
        running_var = conv.bn.running_var
        gamma = conv.bn.weight
        beta = conv.bn.bias
        eps = conv.bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (bias - running_mean) * gamma / std

    def switch_to_deploy(self):
        kernel1, bias1 = self._fuse_bn_tensor(self.conv1)
        kernel2, bias2 = self._fuse_bn_tensor(self.conv2)
        self.conv = self.conv1.conv
        self.conv.weight.data = torch.matmul(kernel2.transpose(1, 3), kernel1.squeeze(3).squeeze(2)).transpose(1, 3)
        self.conv.bias.data = bias2 + (bias1.view(1, -1, 1, 1) * kernel2).sum(3).sum(2).sum(1)
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        self.deploy = True


class RB(nn.Module):
    """Reparameterizable Block.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple): Stride of the convolution. Default: 1
        padding (int, tuple): Padding added to all four sides of
            the input. Default: 1
        padding_mode (string, optional): Default: 'zeros'
        norm_cfg (dict): Config dict to build norm layer.
            Default: dict(type='BN', requires_grad=True)
        act (bool) : Whether to use activation function.
            Default: False
        deploy (bool): Whether in deploy mode. Default: False
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int]] = 3,
                 stride: Union[int, Tuple[int]] = 1,
                 padding: Union[int, Tuple[int]] = 1,
                 padding_mode: Optional[str] = 'zeros',
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True, momentum=0.03, eps=0.001),
                 act: bool = True,
                 deploy: bool = False):
        super().__init__()

        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        if act:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = nn.Identity()

        if deploy:
            self.reparam_3x3 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=True,
                padding_mode=padding_mode)

        else:
            if (out_channels == in_channels) and stride == 1:
                self.path_residual = build_norm_layer(norm_cfg, num_features=in_channels)[1]
            else:
                self.path_residual = None

            self.path_3x3 = ConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=None)
            self.path_1x1 = Block1x1(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                padding=padding_11,
                bias=True,
                norm_cfg=norm_cfg,
            )

    def forward(self, inputs: Tensor) -> Tensor:

        if hasattr(self, 'reparam_3x3'):
            return self.relu(self.reparam_3x3(inputs))

        if self.path_residual is None:
            id_out = 0
        else:
            id_out = self.path_residual(inputs)

        return self.relu(self.path_3x3(inputs) + self.path_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        """Derives the equivalent kernel and bias in a differentiable way.

        Returns:
            tuple: Equivalent kernel and bias
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.path_3x3)
        self.path_1x1.switch_to_deploy()
        kernel1x1, bias1x1 = self.path_1x1.conv.weight.data, self.path_1x1.conv.bias.data
        kernelid, biasid = self._fuse_bn_tensor(self.path_residual)

        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pad 1x1 tensor to 3x3.
        Args:
            kernel1x1 (Tensor): The input 1x1 kernel need to be padded.

        Returns:
            Tensor: 3x3 kernel after padded.
        """
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, conv: nn.Module) -> Tuple[np.ndarray, Tensor]:
        """Derives the equivalent kernel and bias of a specific conv layer.

        Args:
            conv (nn.Module): The layer that needs to be equivalently
                transformed, which can be nn.Sequential or nn.Batchnorm2d

        Returns:
            tuple: Equivalent kernel and bias
        """
        if conv is None:
            return 0, 0
        if isinstance(conv, ConvModule):
            kernel = conv.conv.weight
            running_mean = conv.bn.running_mean
            running_var = conv.bn.running_var
            gamma = conv.bn.weight
            beta = conv.bn.bias
            eps = conv.bn.eps
        else:
            assert isinstance(conv, (nn.SyncBatchNorm, nn.BatchNorm2d))
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3),
                                        dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(
                    conv.weight.device)
            kernel = self.id_tensor
            running_mean = conv.running_mean
            running_var = conv.running_var
            gamma = conv.weight
            beta = conv.bias
            eps = conv.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        """Switch to deploy mode."""
        if hasattr(self, 'reparam_3x3'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.reparam_3x3 = nn.Conv2d(
            in_channels=self.path_3x3.conv.in_channels,
            out_channels=self.path_3x3.conv.out_channels,
            kernel_size=self.path_3x3.conv.kernel_size,
            stride=self.path_3x3.conv.stride,
            padding=self.path_3x3.conv.padding,
            bias=True)
        self.reparam_3x3.weight.data = kernel
        self.reparam_3x3.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('path_3x3')
        self.__delattr__('path_1x1')
        if hasattr(self, 'path_residual'):
            self.__delattr__('path_residual')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True
