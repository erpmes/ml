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

# 加载权重
weights = {}
for name, param in model.named_parameters():
    with open(f'./model2/{name}.txt', 'r') as f:
        lines = f.readlines()
        values = [float(line.strip()) for line in lines]
        weights[name] = torch.tensor(values).reshape(param.shape)

model.load_state_dict(weights)

# 保存模型
torch.save(model.state_dict(), './model2/model.ckpt')

