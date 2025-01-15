import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from nltk.translate.bleu_score import corpus_bleu
from torch.nn.utils.rnn import pad_sequence
import random

# 定义超参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
max_token_length = 50
embedding_size = 256
hidden_size = 512
num_layers = 6
num_heads = 8
dropout = 0.1
num_epochs = 10

# 自定义数据集类
class TranslationDataset(Dataset):
    def __init__(self, en_file, zh_file):
        self.en_sentences = self.read_file(en_file)
        self.zh_sentences = self.read_file(zh_file)
        self.vocab_size = self.build_vocab()
        self.en_token2idx, self.en_idx2token = self.build_token_dicts(self.en_sentences)
        self.zh_token2idx, self.zh_idx2token = self.build_token_dicts(self.zh_sentences)
        self.en_inputs, self.zh_outputs = self.process_data()

    def read_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            sentences = f.read().splitlines()
        return sentences

    def build_vocab(self):
        vocab = set()
        for sentence in self.en_sentences + self.zh_sentences:
            vocab.update(sentence.split())
        return len(vocab)

    def build_token_dicts(self, sentences):
        token2idx = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        idx2token = {0: '<PAD>', 1: '<UNK>', 2: '<SOS>', 3: '<EOS>'}
        for sentence in sentences:
            for token in sentence.split():
                if token not in token2idx:
                    token2idx[token] = len(token2idx)
                    idx2token[len(idx2token)] = token
        return token2idx, idx2token

    def process_data(self):
        en_inputs = []
        zh_outputs = []
        for en_sentence, zh_sentence in zip(self.en_sentences, self.zh_sentences):
            en_inputs.append([self.en_token2idx.get(token, self.en_token2idx['<UNK>']) for token in en_sentence.split()])
            zh_outputs.append([self.zh_token2idx['<SOS>']] + [self.zh_token2idx.get(token, self.zh_token2idx['<UNK>']) for token in zh_sentence.split()] + [self.zh_token2idx['<EOS>']])
        return en_inputs, zh_outputs

    def __len__(self):
        return len(self.en_sentences)

    def __getitem__(self, index):
        return self.en_inputs[index], self.zh_outputs[index]

def collate_fn(batch):
    en_inputs, zh_outputs = zip(*batch)
    en_inputs = pad_sequence([torch.LongTensor(en_input[:max_token_length]) for en_input in en_inputs], batch_first=True, padding_value=0)
    zh_outputs = pad_sequence([torch.LongTensor(zh_output[:max_token_length]) for zh_output in zh_outputs], batch_first=True, padding_value=0)
    return en_inputs.transpose(0, 1), zh_outputs.transpose(0, 1)

# 自定义模型类
class Transformer(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, num_heads, dropout):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(embedding_size, num_heads, hidden_size, dropout), num_layers)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(embedding_size, num_heads, hidden_size, dropout), num_layers)
        self.fc = nn.Linear(embedding_size, vocab_size)

    def forward(self, en_inputs, zh_outputs):
        en_embedded = self.embedding(en_inputs)
        zh_embedded = self.embedding(zh_outputs)
        en_encoded = self.encoder(en_embedded)
        zh_decoded = self.decoder(zh_embedded, en_encoded)
        output = self.fc(zh_decoded)
        return output

# 定义训练函数
def train(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    for en_inputs, zh_outputs in dataloader:
        en_inputs = en_inputs.to(device)
        zh_outputs = zh_outputs.to(device)
        optimizer.zero_grad()
        output = model(en_inputs, zh_outputs[:, :-1])
        loss = criterion(output.view(-1, output.size(2)), zh_outputs[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# 定义验证函数
def evaluate(model, dataloader):
    model.eval()
    references = []
    translations = []
    with torch.no_grad():
        for en_inputs, zh_outputs in dataloader:
            en_inputs = en_inputs.to(device)
            zh_outputs = zh_outputs.to(device)
            output = model(en_inputs, zh_outputs[:, :-1])
            _, predicted = torch.max(output, dim=2)
            references.extend([[ref] for ref in zh_outputs[:, 1:].tolist()])
            translations.extend(predicted.tolist())
    bleu_score = corpus_bleu(references, translations)
    return bleu_score, references, translations

# 加载数据集
train_dataset = TranslationDataset('./DATA/train.clean.en', './DATA/train.clean.zh')
dev_dataset = TranslationDataset('./DATA/valid.clean.en', './DATA/valid.clean.zh')
test_dataset = TranslationDataset('./DATA/test.raw.clean.en', './DATA/test.raw.clean.zh')


# 创建数据加载器
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

# 创建模型实例
model = Transformer(train_dataset.vocab_size, embedding_size, hidden_size, num_layers, num_heads, dropout).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.zh_token2idx['<PAD>'])
optimizer = optim.Adam(model.parameters())

# 训练模型
for epoch in range(num_epochs):
    random.shuffle(train_dataset.en_inputs)
    random.shuffle(train_dataset.zh_outputs)
    train_loss = train(model, train_dataloader, criterion, optimizer)
    dev_bleu, dev_references, dev_translations = evaluate(model, dev_dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Dev BLEU: {dev_bleu:.4f}")

# 在测试集上评估模型
test_bleu, test_references, test_translations = evaluate(model, test_dataloader)
print(f"Test BLEU: {test_bleu:.4f}")

# 将英文和中文对照结果输出到txt文件
with open('test_results.txt', 'w', encoding='utf-8') as f:
    for ref, trans in zip(test_references, test_translations):
        en_ref = ' '.join([test_dataset.en_idx2token[idx] for idx in ref[0]])
        zh_trans = ' '.join([test_dataset.zh_idx2token[idx] for idx in trans])
        f.write(f"EN: {en_ref}\n")
        f.write(f"ZH: {zh_trans}\n")
        f.write("\n")
