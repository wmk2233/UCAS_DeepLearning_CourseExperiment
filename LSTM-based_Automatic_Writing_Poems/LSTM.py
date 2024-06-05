import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 设置一些超参数
Batch_size = 16
learning_rate = 5e-3
embedding_dim = 128
hidden_dim = 256
epochs = 5
verbose = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
pre_trained_model_path = None
trained_model_path = 'model.pth'
start_words = '湖光秋月两相和'
start_words_acrostic = '轻舟已过万重山'
max_gen_len = 128

# 1.加载数据
def prepareData():
    datas = np.load("tang.npz", allow_pickle=True)
    data = datas['data']
    ix2word = datas['ix2word'].item()
    word2ix = datas['word2ix'].item()
    data = torch.from_numpy(data).long()
    dataloader = DataLoader(data, batch_size=Batch_size, shuffle=True, num_workers=0)
    return dataloader, ix2word, word2ix

# 2.定义诗歌模型
class PoetryModel(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_dim):
        super(PoetryModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=2)
        self.linear = nn.Linear(self.hidden_dim, num_embeddings)

    def forward(self, input, hidden=None):
        seq_len, batch_size = input.size()
        if hidden is None:
            h_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        embeds = self.embedding(input)
        output, hidden = self.lstm(embeds, (h_0, c_0))
        output = self.linear(output.view(seq_len * batch_size, -1))
        return output, hidden

# 3.定义训练函数
def train(dataloader, ix2word, word2ix):
    model = PoetryModel(len(word2ix), embedding_dim, hidden_dim).to(device)
    if pre_trained_model_path:
        model.load_state_dict(torch.load(pre_trained_model_path, map_location=device))
        # model.load_state_dict(torch.load(pre_trained_model_path))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        # 计算每个epoch 5%的批次数，至少为1，以避免因为批次太少而导致的除以0的情况
        five_percent_batches = max(len(dataloader) // 20, 1)

        for batch_idx, data in enumerate(dataloader):
            data = data.long().transpose(1, 0).contiguous().to(device)
            input, target = data[:-1, :], data[1:, :]
            optimizer.zero_grad()
            output, _ = model(input)
            loss = criterion(output.view(-1, len(ix2word)), target.view(-1))
            loss.backward()
            optimizer.step()

            # 每5%的批次输出一次训练信息
            if verbose and (batch_idx + 1) % five_percent_batches == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch + 1, (batch_idx + 1) * Batch_size, len(dataloader.dataset),
                    100. * (batch_idx + 1) / len(dataloader), loss.item()))

    torch.save(model.state_dict(), trained_model_path)
    print(f"Training complete. Model saved to {trained_model_path}.")


# 4.定义生成诗歌函数
def generate(start_words, ix2word, word2ix):
    model = PoetryModel(len(word2ix), embedding_dim, hidden_dim).to(device)
    model.load_state_dict(torch.load(trained_model_path))
    model.to(device)

    results = list(start_words)
    start_word_len = len(start_words)
    input = torch.Tensor([word2ix['<START>']]).view(1, 1).long().to(device)
    hidden = None

    for i in range(max_gen_len):
        output, hidden = model(input, hidden)
        if i < start_word_len:
            w = results[i]
            input = input.data.new([word2ix[w]]).view(1, 1)
        else:
            top_index = output.data.cpu().numpy().argmax()
            w = ix2word[top_index]
            results.append(w)
            input = input.data.new([top_index]).view(1, 1)
        if w == '<EOP>':
            del results[-1]
            break
    return ''.join(results)

# 5.定义生成藏头诗函数
def gen_acrostic(start_words, ix2word, word2ix):
    model = PoetryModel(len(word2ix), embedding_dim, hidden_dim).to(device)
    model.load_state_dict(torch.load(trained_model_path))
    model.to(device)

    results = []
    start_word_len = len(start_words)
    input = torch.Tensor([word2ix['<START>']]).view(1, 1).long().to(device)
    hidden = None
    index = 0
    pre_word = '<START>'

    for i in range(max_gen_len):
        output, hidden = model(input, hidden)
        top_index = output.data.cpu().numpy().argmax()
        w = ix2word[top_index]

        if pre_word in {u'。', u'！', '<START>'}:
            if index == start_word_len:
                break
            else:
                w = start_words[index]
                index += 1
                input = input.data.new([word2ix[w]]).view(1, 1)
        else:
            input = input.data.new([top_index]).view(1, 1)
        results.append(w)
        pre_word = w
    return ''.join(results)


def format_poem_with_newlines(poem_text):
    # 在每个句号后面添加换行符（注意：句号后面可能已经有空格，因此先去除空格再添加换行符）
    formatted_poem = poem_text.replace("。", "。\n").strip()
    return formatted_poem
# 准备数据和训练模型
dataloader, ix2word, word2ix = prepareData()
train(dataloader, ix2word, word2ix)

# 生成诗歌并格式化输出
poetry = generate(start_words, ix2word, word2ix)
formatted_poetry = format_poem_with_newlines(poetry)
print('自动写诗：\n', formatted_poetry)

# 生成藏头诗并格式化输出
acrostic = gen_acrostic(start_words_acrostic, ix2word, word2ix)
formatted_acrostic = format_poem_with_newlines(acrostic)
print('藏头诗：\n', formatted_acrostic)
