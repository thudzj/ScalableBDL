import torch
import torch.nn as nn
from torch.nn import init
from mean_field import *
from functools import partial

class Block(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropout_rate, conv_layer, bn_layer):
        super(Block, self).__init__()

        self.bn1 = bn_layer(in_planes)
        self.conv1 = conv_layer(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = bn_layer(out_planes)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.conv2 = conv_layer(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and conv_layer(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False) or None

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(x))
        if not self.equalInOut: residual = out
        out = self.conv2(self.dropout(self.relu(self.bn2(self.conv1(out)))))
        if self.convShortcut is not None: residual = self.convShortcut(residual)
        return out + residual

class WRN(nn.Module):
    def __init__(self, depth, width, num_classes, args):
        super(WRN, self).__init__()

        n_channels = [16, 16*width, 32*width, 64*width]
        assert((depth - 4) % 6 == 0)
        num_blocks = (depth - 4) // 6
        # print ('WRN : Depth : {} , Widen Factor : {}'.format(depth, width))

        self.num_classes = num_classes
        self.bayes = args.bayes

        if self.bayes is None:
            conv_layer, linear_layer, bn_layer = nn.Conv2d, nn.Linear, nn.BatchNorm2d
        elif self.bayes == 'mf':
            conv_layer, linear_layer, bn_layer = partial(BayesConv2dMF, args.single_eps, args.local_reparam), partial(BayesLinearMF, args.single_eps, args.local_reparam), partial(BayesBatchNorm2dMF, args.single_eps)
        else:
            raise NotImplementedError


        self.conv_3x3 = conv_layer(3, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)

        self.stage_1 = self._make_layer(n_channels[0], n_channels[1], num_blocks, 1, args.dropout_rate, conv_layer, bn_layer)

        self.stage_2 = self._make_layer(n_channels[1], n_channels[2], num_blocks, 2, args.dropout_rate, conv_layer, bn_layer)

        self.stage_3 = self._make_layer(n_channels[2], n_channels[3], num_blocks, 2, args.dropout_rate, conv_layer, bn_layer)

        self.lastact = nn.Sequential(bn_layer(n_channels[3]), nn.ReLU(inplace=True))
        self.avgpool = nn.AvgPool2d(8)
        self.classifier = linear_layer(n_channels[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, BayesConv2dMF):
                init.kaiming_normal_(m.weight_mu)
                m.weight_log_sigma.data.uniform_(args.log_sigma_init_range[0], args.log_sigma_init_range[1])
            elif isinstance(m, BayesBatchNorm2dMF):
                m.weight_mu.data.fill_(1)
                m.bias_mu.data.zero_()
                m.weight_log_sigma.data.uniform_(args.log_sigma_init_range[0], args.log_sigma_init_range[1])
                m.bias_log_sigma.data.uniform_(args.log_sigma_init_range[0], args.log_sigma_init_range[1])
            elif isinstance(m, BayesLinearMF):
                init.kaiming_normal_(m.weight_mu)
                m.bias_mu.data.zero_()
                m.weight_log_sigma.data.uniform_(args.log_sigma_init_range[0], args.log_sigma_init_range[1])
                m.bias_log_sigma.data.uniform_(args.log_sigma_init_range[0], args.log_sigma_init_range[1])

    def _make_layer(self, in_planes, out_planes, num_blocks, stride, dropout_rate, conv_layer, bn_layer):
        blocks = []
        blocks.append(Block(in_planes, out_planes, stride, dropout_rate, conv_layer, bn_layer))
        for i in range(1, num_blocks): blocks.append(Block(out_planes, out_planes, 1, dropout_rate, conv_layer,bn_layer))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.conv_3x3(x)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.lastact(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def wrn(args, depth, width, num_classes=10):
    model = WRN(depth, width, num_classes, args)
    return model
