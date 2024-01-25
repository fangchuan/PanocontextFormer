import torch.nn as nn
import torch.nn.functional as F

BN_MOMENTUM = 0.1

class global_feature_project(nn.Module):
    def __init__(self, global_featrue_dim=128):
        super(global_feature_project, self).__init__()
        self.global_featrue_dim = global_featrue_dim

        self.downsamp_modules, self.final_layer = self._make_head(self.global_featrue_dim)

    def _make_head(self, global_featrue_dim):
        head_channels = [64, 128, 256, 512]

        # downsampling modules
        downsamp_modules = []
        for i in range(len(head_channels) - 1):
            in_channels = head_channels[i]
            out_channels = head_channels[i + 1]

            downsamp_module = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=3,
                          stride=2,
                          padding=1),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            )

            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=head_channels[3],
                out_channels=global_featrue_dim,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.BatchNorm2d(global_featrue_dim, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )

        return downsamp_modules, final_layer

    def forward(self, y_list):

        for i in range(len(self.downsamp_modules)):
            if (i == 0):
                y = self.downsamp_modules[i](y_list[i])
            else:
                y = self.downsamp_modules[i](y_list[i] + y)
        y = self.final_layer(y_list[3] + y)
        y = F.avg_pool2d(y, kernel_size=y.size()[2:]).view(y.size(0), -1)

        return y

class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class global_feature_project2(nn.Module):
    def __init__(self, global_feature_dim = 128):
        super(global_feature_project2, self).__init__()

        self.global_featrue_dim = global_feature_dim

        self.incre_modules, self.downsamp_modules, self.final_layer = self._make_head([64, 128, 256, 512], self.global_featrue_dim)

        self.init_weight()

    def forward(self, y_list):
        # Classification Head
        y = self.incre_modules[0](y_list[0])
        for i in range(len(self.downsamp_modules)):
            y = self.incre_modules[i+1](y_list[i+1]) + \
                        self.downsamp_modules[i](y)

        y = self.final_layer(y)
        y = F.avg_pool2d(y, kernel_size=y.size()[2:]).view(y.size(0), -1)

        return y

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_head(self, pre_stage_channels, global_featrue_dim):

        BN_MOMENTUM = 0.1
        head_block = Bottleneck
        head_channels = [32, 64, 128, 256]

        # Increasing the #channels on each resolution
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024
        incre_modules = []
        for i, channels in enumerate(pre_stage_channels):
            incre_module = self._make_layer(head_block,
                                            channels,
                                            head_channels[i],
                                            1,
                                            stride=1)
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)

        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            in_channels = head_channels[i] * head_block.expansion
            out_channels = head_channels[i + 1] * head_block.expansion

            downsamp_module = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=3,
                          stride=2,
                          padding=1),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            )

            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=head_channels[3] * head_block.expansion,
                out_channels=global_featrue_dim,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.BatchNorm2d(global_featrue_dim, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )

        return incre_modules, downsamp_modules, final_layer

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)