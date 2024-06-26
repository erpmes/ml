# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

# Neural Network Model
class My_Model(nn.Module):
    def __init__(self, input_dim):
        super(My_Model, self).__init__()
        # TODO: modify model's structure, be aware of dimensions. 
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1) # (B, 1) -> (B)
        return x

# 创建模型实例
model = My_Model(117)

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

