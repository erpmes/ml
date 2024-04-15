import torchimport torch.nn as nn### Define Modelclass Classifier(nn.Module):  def __init__(self, d_model=80, n_spks=600, dropout=0.1):    super().__init__()    # Project the dimension of features from that of input into d_model.    self.prenet = nn.Linear(40, d_model)    # TODO:    #   Change Transformer to Conformer.    #   https://arxiv.org/abs/2005.08100    self.encoder_layer = nn.TransformerEncoderLayer(      d_model=d_model, dim_feedforward=256, nhead=2    )    # self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)    # Project the the dimension of features from d_model into speaker nums.    self.pred_layer = nn.Sequential(      nn.Linear(d_model, d_model),      nn.ReLU(),      nn.Linear(d_model, n_spks),    )  def forward(self, mels):    """    args:      mels: (batch size, length, 40)    return:      out: (batch size, n_spks)    """    # out: (batch size, length, d_model)    out = self.prenet(mels)    # out: (length, batch size, d_model)    out = out.permute(1, 0, 2)    # The encoder layer expect features in the shape of (length, batch size, d_model).    out = self.encoder_layer(out)    # out: (batch size, length, d_model)    out = out.transpose(0, 1)    # mean pooling    stats = out.mean(dim=1)    # out: (batch, n_spks)    out = self.pred_layer(stats)    return out# "cuda" only when GPUs are available.#device = "cuda" if torch.cuda.is_available() else "cpu"device ="cpu"# Initialize a model, and put it on the device specified.model = Classifier().to(device)model.load_state_dict(torch.load('./model.ckpt'))import json# 假设你的state_dict是一个名为state_dict的字典state_dict = model.state_dict()# 遍历每一层的state_dict，保存到单独的文件中for layer_name, layer_state in state_dict.items():    # 将state_dict中的Tensor对象转换为NumPy数组，然后再转换为Python的内置数据类型    layer_state_python = layer_state.numpy().tolist()    # 将layer_state_python转换为JSON格式的字符串    layer_state_json = json.dumps(layer_state_python)    # 将layer_state_json保存到单独的文件中    file_name = f"{layer_name}.txt"    with open(file_name, 'w') as f:        f.write(layer_state_json)'''# 假设你的state_dict是一个名为state_dict的字典state_dict = model.state_dict()# 将state_dict中的Tensor对象转换为NumPy数组，然后再转换为Python的内置数据类型state_dict_python = {}for key, value in state_dict.items():    state_dict_python[key] = value.numpy().tolist()# 将state_dict_python转换为JSON格式的字符串state_dict_json = json.dumps(state_dict_python)# 将state_dict_json保存到文本文件with open('./state_dict.txt', 'w') as f:    f.write(state_dict_json)''''''# 提取模型权重weights = {}for name, param in model.named_parameters():    weights[name] = param.data.numpy()# 保存权重为txt文件for name, value in weights.items():    with open(f'./{name}.txt', 'w') as f:        f.write('\n'.join(value.flatten().astype(str)))'''