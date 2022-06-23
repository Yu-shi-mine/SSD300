"""
SSDのextrasモジュール
"""

import torch.nn as nn

def extras(cfg = [256, 512, 128, 256, 128, 256, 128, 256]):
    """_summary_

    Args:
        cfg (list, optional): extraモジュールの畳み込み層のチャネル数を設定するコンフィギュレーション. Defaults to [256, 512, 128, 256, 128, 256, 128, 256].

    Returns:
        nn.ModuleList: _descSSDのextrasモジュールription_
    """

    layers = []
    in_channels = 1024  # vggモジュールから出力された、extraに入力される画像チャネル数

    layers += [nn.Conv2d(in_channels, cfg[0], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[0], cfg[1], kernel_size=(3), stride=2, padding=1)]
    layers += [nn.Conv2d(cfg[1], cfg[2], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[2], cfg[3], kernel_size=(3), stride=2, padding=1)]
    layers += [nn.Conv2d(cfg[3], cfg[4], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[4], cfg[5], kernel_size=(3))]
    layers += [nn.Conv2d(cfg[5], cfg[6], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[6], cfg[7], kernel_size=(3))]

    return nn.ModuleList(layers)