# wujian@2018
"""
Conv-TasNet 模型定义。

职责：
1. 定义编码器、时域分离主干和解码器；
2. 提供 Conv-TasNet 训练与推理时统一的前向逻辑；
3. 封装本项目使用的归一化层、1D 卷积封装和重复卷积块。

核心输入输出：
- 输入：单通道混合语音，`Tensor[S]` 或 `Tensor[N, S]`
- 输出：说话人分离结果列表，`List[Tensor[N, S']]`
"""

import torch as th
import torch.nn as nn

import torch.nn.functional as F


def param(nnet, Mb=True):
    """
    统计模型参数量。

    输入：
    - nnet: 任意 `nn.Module`
    - Mb: 为 True 时返回百万参数量，否则返回参数总个数

    输出：
    - float 或 int
    """
    neles = sum([param.nelement() for param in nnet.parameters()])
    return neles / 10**6 if Mb else neles


class ChannelWiseLayerNorm(nn.LayerNorm):
    """
    按通道做 LayerNorm。

    输入：
    - `x`: `Tensor[N, C, T]`

    输出：
    - `Tensor[N, C, T]`
    """

    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        前向输入输出形状均为 `N x C x T`。
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # N x C x T => N x T x C
        x = th.transpose(x, 1, 2)
        # LN
        x = super().forward(x)
        # N x C x T => N x T x C
        x = th.transpose(x, 1, 2)
        return x


class GlobalChannelLayerNorm(nn.Module):
    """
    全局通道归一化。

    会在每条样本的 `(C, T)` 全部位置上统计均值和方差。

    输入：
    - `x`: `Tensor[N, C, T]`

    输出：
    - `Tensor[N, C, T]`
    """

    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super(GlobalChannelLayerNorm, self).__init__()
        self.eps = eps
        self.normalized_dim = dim
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.beta = nn.Parameter(th.zeros(dim, 1))
            self.gamma = nn.Parameter(th.ones(dim, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        """
        前向输入输出形状均为 `N x C x T`。
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # N x 1 x 1
        mean = th.mean(x, (1, 2), keepdim=True)
        var = th.mean((x - mean)**2, (1, 2), keepdim=True)
        # N x T x C
        if self.elementwise_affine:
            x = self.gamma * (x - mean) / th.sqrt(var + self.eps) + self.beta
        else:
            x = (x - mean) / th.sqrt(var + self.eps)
        return x

    def extra_repr(self):
        return "{normalized_dim}, eps={eps}, " \
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)


def build_norm(norm, dim):
    """
    根据名称构造归一化层。

    输入：
    - norm: `"cLN"` / `"gLN"` / `"BN"`
    - dim: 通道数

    输出：
    - 对应的归一化层实例
    """
    if norm not in ["cLN", "gLN", "BN"]:
        raise RuntimeError("Unsupported normalize layer: {}".format(norm))
    if norm == "cLN":
        return ChannelWiseLayerNorm(dim, elementwise_affine=True)
    elif norm == "BN":
        return nn.BatchNorm1d(dim)
    else:
        return GlobalChannelLayerNorm(dim, elementwise_affine=True)


class Conv1D(nn.Conv1d):
    """
    对 `nn.Conv1d` 的轻量封装。

    兼容两种输入：
    - `Tensor[N, L]`
    - `Tensor[N, C, L]`

    输出：
    - 默认返回三维张量
    - `squeeze=True` 时去掉单通道维
    """

    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        """
        输入形状：
        - `N x L` 或 `N x C x L`
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else th.unsqueeze(x, 1))
        if squeeze:
            x = th.squeeze(x)
        return x


class ConvTrans1D(nn.ConvTranspose1d):
    """
    对 `nn.ConvTranspose1d` 的轻量封装。

    输入：
    - `Tensor[N, L]` 或 `Tensor[N, C, L]`

    输出：
    - 反卷积后的时域序列
    """

    def __init__(self, *args, **kwargs):
        super(ConvTrans1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        """
        输入形状：
        - `N x L` 或 `N x C x L`
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else th.unsqueeze(x, 1))
        if squeeze:
            x = th.squeeze(x)
        return x


class Conv1DBlock(nn.Module):
    """
    Conv-TasNet 中的基础卷积块。

    结构：
    `1x1 Conv -> PReLU -> Norm -> Depthwise Dilated Conv
     -> PReLU -> Norm -> 1x1 Conv -> Residual Add`

    输入输出：
    - 输入：`Tensor[N, B, T]`
    - 输出：`Tensor[N, B, T]`
    """

    def __init__(self,
                 in_channels=256,
                 conv_channels=512,
                 kernel_size=3,
                 dilation=1,
                 norm="cLN",
                 causal=False):
        super(Conv1DBlock, self).__init__()
        # 1x1 conv
        self.conv1x1 = Conv1D(in_channels, conv_channels, 1)
        self.prelu1 = nn.PReLU()
        self.lnorm1 = build_norm(norm, conv_channels)
        dconv_pad = (dilation * (kernel_size - 1)) // 2 if not causal else (
            dilation * (kernel_size - 1))
        # depthwise conv
        self.dconv = nn.Conv1d(
            conv_channels,
            conv_channels,
            kernel_size,
            groups=conv_channels,
            padding=dconv_pad,
            dilation=dilation,
            bias=True)
        self.prelu2 = nn.PReLU()
        self.lnorm2 = build_norm(norm, conv_channels)
        # 1x1 conv cross channel
        self.sconv = nn.Conv1d(conv_channels, in_channels, 1, bias=True)
        # different padding way
        self.causal = causal
        self.dconv_pad = dconv_pad

    def forward(self, x):
        y = self.conv1x1(x)
        y = self.lnorm1(self.prelu1(y))
        y = self.dconv(y)
        if self.causal:
            y = y[:, :, :-self.dconv_pad]
        y = self.lnorm2(self.prelu2(y))
        y = self.sconv(y)
        x = x + y
        return x


class ConvTasNet(nn.Module):
    """
    Conv-TasNet 主模型。

    流程：
    1. 编码器把混合波形变成时域特征；
    2. TCN 风格的重复卷积块估计每个说话人的 mask；
    3. 用 mask 作用在编码特征上；
    4. 解码器把每路特征还原成波形。

    输入：
    - `Tensor[S]` 或 `Tensor[N, S]`

    输出：
    - `List[Tensor[N, S']]`，列表长度等于 `num_spks`
    """

    def __init__(self,
                 L=20,
                 N=256,
                 X=8,
                 R=4,
                 B=256,
                 H=512,
                 P=3,
                 norm="cLN",
                 num_spks=2,
                 non_linear="relu",
                 causal=False):
        super(ConvTasNet, self).__init__()
        supported_nonlinear = {
            "relu": F.relu,
            "sigmoid": th.sigmoid,
            "softmax": F.softmax
        }
        if non_linear not in supported_nonlinear:
            raise RuntimeError("Unsupported non-linear function: {}",
                               format(non_linear))
        self.non_linear_type = non_linear
        self.non_linear = supported_nonlinear[non_linear]
        # n x S => n x N x T, S = 4s*8000 = 32000
        self.encoder_1d = Conv1D(1, N, L, stride=L // 2, padding=0)
        # keep T not change
        # T = int((xlen - L) / (L // 2)) + 1
        # before repeat blocks, always cLN
        self.ln = ChannelWiseLayerNorm(N)
        # n x N x T => n x B x T
        self.proj = Conv1D(N, B, 1)
        # repeat blocks
        # n x B x T => n x B x T
        self.repeats = self._build_repeats(
            R,
            X,
            in_channels=B,
            conv_channels=H,
            kernel_size=P,
            norm=norm,
            causal=causal)
        # output 1x1 conv
        # n x B x T => n x N x T
        # NOTE: using ModuleList not python list
        # self.conv1x1_2 = th.nn.ModuleList(
        #     [Conv1D(B, N, 1) for _ in range(num_spks)])
        # n x B x T => n x 2N x T
        self.mask = Conv1D(B, num_spks * N, 1)
        # using ConvTrans1D: n x N x T => n x 1 x To
        # To = (T - 1) * L // 2 + L
        self.decoder_1d = ConvTrans1D(
            N, 1, kernel_size=L, stride=L // 2, bias=True)
        self.num_spks = num_spks

    def _build_blocks(self, num_blocks, **block_kwargs):
        """
        构造一组不同 dilation 的卷积块。
        """
        blocks = [
            Conv1DBlock(**block_kwargs, dilation=(2**b))
            for b in range(num_blocks)
        ]
        return nn.Sequential(*blocks)

    def _build_repeats(self, num_repeats, num_blocks, **block_kwargs):
        """
        构造重复堆叠的卷积块主干。
        """
        repeats = [
            self._build_blocks(num_blocks, **block_kwargs)
            for r in range(num_repeats)
        ]
        return nn.Sequential(*repeats)

    def forward(self, x):
        """
        Conv-TasNet 前向传播。

        输入：
        - `x`: `Tensor[S]` 或 `Tensor[N, S]`

        输出：
        - `List[Tensor[N, S']]`
        - 每个元素对应一路分离出来的说话人波形
        """
        if x.dim() >= 3:
            raise RuntimeError(
                "{} accept 1/2D tensor as input, but got {:d}".format(
                    self.__name__, x.dim()))
        # when inference, only one utt
        if x.dim() == 1:
            x = th.unsqueeze(x, 0)
        # 编码阶段：把时域波形映射到可学习的基底表示。
        # n x 1 x S => n x N x T
        w = F.relu(self.encoder_1d(x))
        # 归一化并投影到 bottleneck 通道。
        # n x B x T
        y = self.proj(self.ln(w))
        # 分离主干：堆叠的时域卷积块负责建模说话人上下文。
        # n x B x T
        y = self.repeats(y)
        # 掩码估计：为每位说话人预测一份编码域 mask。
        # n x 2N x T
        e = th.chunk(self.mask(y), self.num_spks, 1)
        # 根据配置选择激活函数，得到真正的 mask。
        # n x N x T
        if self.non_linear_type == "softmax":
            m = self.non_linear(th.stack(e, dim=0), dim=0)
        else:
            m = self.non_linear(th.stack(e, dim=0))
        # 将每位说话人的 mask 作用到共享编码特征上。
        # spks x [n x N x T]
        s = [w * m[n] for n in range(self.num_spks)]
        # 解码阶段：把每一路编码特征还原回时域波形。
        # spks x n x S
        return [self.decoder_1d(x, squeeze=True) for x in s]


def foo_conv1d_block():
    nnet = Conv1DBlock(256, 512, 3, 20)
    print(param(nnet))


def foo_layernorm():
    C, T = 256, 20
    nnet1 = nn.LayerNorm([C, T], elementwise_affine=True)
    print(param(nnet1, Mb=False))
    nnet2 = nn.LayerNorm([C, T], elementwise_affine=False)
    print(param(nnet2, Mb=False))


def foo_conv_tas_net():
    x = th.rand(4, 1000)
    nnet = ConvTasNet(norm="cLN", causal=False)
    # print(nnet)
    print("ConvTasNet #param: {:.2f}".format(param(nnet)))
    x = nnet(x)
    s1 = x[0]
    print(s1.shape)


if __name__ == "__main__":
    foo_conv_tas_net()
    # foo_conv1d_block()
    # foo_layernorm()
