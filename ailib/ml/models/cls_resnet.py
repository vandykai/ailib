import torch
import torch.nn as nn
from ailib.models.base_model_param import BaseModelParam
from ailib.models.base_model import BaseModel
from ailib.param.param import Param
from ailib.param import hyper_spaces
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class ModelParam(BaseModelParam):

    def __init__(self, with_embedding=False, with_multi_layer_perceptron=False):
        super().__init__(with_embedding, with_multi_layer_perceptron)
        self['model_name'] = "CNN"
        self['learning_rate'] = 1e-3
        self.add(Param(name='input_channel', value=1, desc="input channel"))
        self.add(Param(name='input_dim', value=16384, desc="input dim"))
        self.add(Param(
            name='blocks', value=[3, 4, 6, 3],
            hyper_space=hyper_spaces.choice(
                options=[[3, 4, 6, 3], [3, 4, 23, 3], [3, 8, 36, 3]]),
            desc="ResNet50, ResNet101, ResNet152"))

def ConvHead(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes, out_channels=places, kernel_size=7, stride=stride, padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

class Bottleneck(nn.Module):
    def __init__(self, in_places, places, stride=1, downsampling=False, expansion = 4):
        super().__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places*self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        residual = inputs
        out = self.bottleneck(inputs)

        if self.downsampling:
            residual = self.downsample(inputs)

        out += residual
        out = self.relu(out)
        return out

class Model(BaseModel):
    def __init__(self, config, expansion=4):
        super().__init__()
        self.config = config
        self.expansion = expansion
        self.preprocess = nn.Sequential(
            Rearrange('b w h -> b 1 (w h)'),
            nn.Linear(config.input_dim, config.input_dim),
            nn.LayerNorm(config.input_dim)
        )
        self.conv1 = ConvHead(in_planes=config.input_channel, places=64)

        self.layer1 = self._make_layer(in_places=64, places=64, block=config.blocks[0], stride=1)
        self.layer2 = self._make_layer(in_places=256,places=128, block=config.blocks[1], stride=2)
        self.layer3 = self._make_layer(in_places=512, places=256, block=config.blocks[2], stride=2)
        self.layer4 = self._make_layer(in_places=1024, places=512, block=config.blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = self._make_output_layer(2048)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places,stride, downsampling =True))
        for i in range(1, block):
            layers.append(Bottleneck(places*self.expansion, places))

        return nn.Sequential(*layers)


    def forward(self, inputs):
        device = inputs['device']
        # [batch_size, number]
        # [N, 1, H, W]
        feature_map = torch.sparse_coo_tensor(inputs['indices'], inputs['values'], inputs['shape'], device=device).to_dense()
        *_, w, h = feature_map.shape
        feature_map = self.preprocess(feature_map)
        feature_map = rearrange(feature_map, 'b c (w h) -> b c w h', w = w, h = h)
        out = self.conv1(feature_map)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
