# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
### Define Model

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 8, 8]
            
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Linear(512*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)    

# "cuda" only when GPUs are available.
#device = "cuda" if torch.cuda.is_available() else "cpu"
device ="cpu"


# Initialize a model, and put it on the device specified.
model = Classifier().to(device)


import json
import numpy as np

# 获取模型的state_dict
state_dict = model.state_dict()
# 获取模型的层名称
layer_names = list(state_dict.keys())

# 遍历每个层的state_dict文件，加载到state_dict字典中
for layer_name in layer_names:
    # 读取layer_name.txt文件
    with open(f"{layer_name}.txt", 'r') as f:
        layer_state_json = f.read()

    # 将layer_state_json解析为Python的内置数据类型
    layer_state_python = json.loads(layer_state_json)

    # 将layer_state_python中的Python内置数据类型转换回NumPy数组，然后再转换回Tensor对象
    layer_state = torch.from_numpy(np.array(layer_state_python))

    # 将layer_state加载到state_dict字典中
    state_dict[layer_name] = layer_state

# 将state_dict加载到模型中
model.load_state_dict(state_dict)

'''
# 读取state_dict文本文件
with open('./state_dict.txt', 'r') as f:
    state_dict_json = f.read()

# 将state_dict_json解析为字典
state_dict_python = json.loads(state_dict_json)

# 将state_dict_python中的Python内置数据类型转换回NumPy数组，然后再转换回Tensor对象
state_dict = {}
for key, value in state_dict_python.items():
    state_dict[key] = torch.from_numpy(np.array(value))

# 将解析后的state_dict加载到模型中
model.load_state_dict(state_dict)
'''

'''
# 加载权重
weights = {}
for name, param in model.named_parameters():
    with open(f'./{name}.txt', 'r') as f:
        lines = f.readlines()
        values = [float(line.strip()) for line in lines]
        weights[name] = torch.tensor(values).reshape(param.shape)

model.load_state_dict(weights)
'''

# 保存模型
torch.save(model.state_dict(), './model.ckpt')

