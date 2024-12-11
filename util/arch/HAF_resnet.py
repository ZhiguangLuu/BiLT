import torch
import torch.nn as nn
from torchvision import models


def model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print('model size (param + buffer): {:.3f}MB'.format(size_all_mb))
    return size_all_mb


def model_size_verbose(model):
    param_size = 0
    for name, param in model.named_parameters():
        print(f"add {name}: {param.nelement() * param.element_size() / 1024 ** 2 :.4f} MB")
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print('model size (param + buffer): {:.3f}MB'.format(size_all_mb))
    return size_all_mb


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
        raise ValueError(f"activation function '{activation_function}' unrecognized")


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


class HAFrameResNet50(nn.Module):
    def __init__(self, pooling, model, num_classes, haf_cls_weights=None):
        super(HAFrameResNet50, self).__init__()

        # self.num_ftrs = 512 * 1 * 1       # Used for resnet18
        self.num_ftrs = 2048 * 1 * 1      # Used for resnet50
        backbone = list(model.children())[:-2]
        # add 1x1 conv layer: channel-wise downsampling
        backbone.append(nn.Conv2d(self.num_ftrs, num_classes,
                        kernel_size=1, stride=1, padding=0, bias=False))
        self.features_2 = nn.Sequential(*backbone)

        if pooling == "max":
            self.pool = nn.MaxPool2d(kernel_size=7, stride=7) # pooling
        else:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # flattening, followed by self.features_1 (transformation layer)
        self.features_1 = nn.Sequential(
            nn.BatchNorm1d(num_classes),
            NarrowResidualTransformationHead(
                num_classes,
                num_classes,
                'prelu',
                )
        )

        self.classifier_3 = nn.Linear(num_classes, num_classes, bias=True)
        if haf_cls_weights is not None:
            with torch.no_grad():
                self.classifier_3.weight = nn.Parameter(torch.Tensor(haf_cls_weights))
                self.classifier_3.weight.requires_grad_(False)
                self.classifier_3.bias = nn.Parameter(torch.zeros([num_classes, ]))
                self.classifier_3.bias.requires_grad_(False)

    def forward(self, x, target="ignored"):
        x = self.features_2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.features_1(x) # N * num_classes
        x = self.classifier_3(x)
        return x

    def penultimate_feature(self, x):
        x = self.features_2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.features_1(x) # N * num_classes
        return x


if __name__ == '__main__':
    model = models.resnet50(pretrained=True)
    model = HAFrameResNet50(model, 1010)
    print(model)

    try:
        # x = torch.rand((4, 2048, 56, 56))
        x = torch.rand((4, 3, 224, 224))
        y = model(x)
        print('forward pass succeeded')
    except ValueError:
        print('!!!!!forward pass value error!!!!!')

    print('done')
