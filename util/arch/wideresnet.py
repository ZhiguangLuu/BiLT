import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, activate_before_residual))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, num_classes, depth=28, widen_factor=8, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, activate_before_residual=True)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return out                     # hierarchical wideresnet


# HAFrame model
def get_activation_function(activation_function, input_channels):
    if activation_function == 'relu':
        return nn.ReLU()
    elif activation_function == 'elu':
        return nn.ELU()
    elif activation_function == 'tanh':
        return nn.Tanh()
    elif activation_function == 'prelu':
        return nn.PReLU(num_parameters=input_channels)
    else:
        raise ValueError(f"activation function '{activation_function}' not supported")


class PointResidualTransformationLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, activation_function='relu', dropout=False):
        super(PointResidualTransformationLayer, self).__init__()
        self.in_features = in_channels
        self.hidden_features = hidden_channels
        self.out_features = out_channels
        self.dropout = dropout
        self.linear1 = nn.Linear(in_channels, hidden_channels, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.act_func1 = get_activation_function(activation_function, hidden_channels)
        if dropout: self.dp1 = nn.Dropout(p=0.5, inplace=False)
        self.linear2 = nn.Linear(hidden_channels, out_channels, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act_func2 = get_activation_function(activation_function, out_channels)
        if dropout: self.dp2 = nn.Dropout(p=0.5, inplace=False)

        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Linear(in_channels, out_channels, bias=False),
                # nn.BatchNorm1d(out_channels) # preact doesn't apply bn in short cut
            )

    def forward(self, x):
        # residual connection
        r = self.residual(x)

        x = self.linear1(x)
        x = self.bn1(x)
        x = self.act_func1(x)
        if self.dropout:
            x = self.dp1(x)

        x = self.linear2(x)
        x = self.bn2(x)
        x = self.act_func2(x)
        if self.dropout:
            x = self.dp2(x)

        # summation
        y = x + r
        return y


class NarrowResidualTransformationHead(nn.Module):
    def __init__(self, in_channels, out_channels, activation_layer,
                 dropout=False, res_layer=PointResidualTransformationLayer):
        super(NarrowResidualTransformationHead,self).__init__()
        self.in_features = in_channels
        self.out_features = out_channels
        if isinstance(res_layer, PointResidualTransformationLayer):
            self.residual_layer1 = res_layer(in_channels, out_channels,
                                             out_channels, activation_layer, dropout)
            self.residual_layer2 = res_layer(out_channels, out_channels,
                                             out_channels, activation_layer, dropout)
        else:
            self.residual_layer1 = res_layer(in_channels, out_channels,
                                             out_channels, activation_layer)
            self.residual_layer2 = res_layer(out_channels, out_channels,
                                             out_channels, activation_layer)

    def forward(self, x):
        x = self.residual_layer1(x)
        x = self.residual_layer2(x)
        return x


class HAFrameWideResNet(nn.Module):
    def __init__(self, num_classes, depth=28, widen_factor=8, dropRate=0.0):
        super(HAFrameWideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, activate_before_residual=True)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # added: 1x1 conv layer to downsample channels to num_classes before pooling
        self.downsample = nn.Conv2d(nChannels[3], num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm1d(num_classes, momentum=0.001)
        self.projection = NarrowResidualTransformationHead(num_classes,
                                                           num_classes,
                                                           'prelu')
        # self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fc = nn.Linear(num_classes, num_classes)
        self.nChannels = num_classes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        # add 1x1 conv before pooling
        out = self.downsample(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        # after pooling and flattening, add bn and transformation module
        out = self.bn2(out)
        out = self.projection(out)
        return out


if __name__ == "__main__":
    model = WideResNet(100)
    tp = nn.Sequential(*list(model.children())[:-1])
    print(tp)
