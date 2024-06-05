import torch.nn.functional as F
import math
import torch
from nltk.tokenize import word_tokenize
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import copy
import numpy as np
import os
import re
import sacrebleu
import random
import time
import jieba
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.corpus import words
import nltk
import matplotlib.pyplot as plt

# 下载需要的NLTK数据
nltk.download('words')

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 定义TranslationDataset类
class TranslationDataset(Dataset):
    def __init__(self, src, tgt):
        self.src = src
        self.tgt = tgt

    def __getitem__(self, i):
        return self.src[i], self.tgt[i]

    def __len__(self):
        return len(self.src)

# 定义Tokenizer类
class Tokenizer():
    def __init__(self, en_path, zh_path, count_min=5):
        self.en_path = en_path
        self.zh_path = zh_path
        self.__count_min = count_min
        self.en_data = self.__read_ori_data(en_path)
        self.zh_data = self.__read_ori_data(zh_path)
        self.index_2_word = ['unK', '<pad>', '<bos>', '<eos>']
        self.word_2_index = {'unK': 0, '<pad>': 1, '<bos>': 2, '<eos>': 3}
        self.en_set = set()
        self.en_count = {}
        self.__count_word()
        self.__filter_data()
        random.shuffle(self.data_)
        validation_size = 2000  # 修改验证集大小为2000条
        self.test = self.data_[-validation_size:]  # 取最后2000条数据作为验证集
        self.data_ = self.data_[:-validation_size]  # 剩余数据作为训练集

    def __read_ori_data(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = f.read().split('\n')[:-1]
        return data

    def __count_word(self):
        le = len(self.en_data)
        p = 0
        for data in self.en_data:
            if p % 1000 == 0:
                print('英文', p / le)
            sentence = word_tokenize(data)
            for sen in sentence:
                if sen in self.en_set:
                    self.en_count[sen] += 1
                else:
                    self.en_set.add(sen)
                    self.en_count[sen] = 1
            p += 1
        for k, v in self.en_count.items():
            if v >= self.__count_min:
                self.word_2_index[k] = len(self.index_2_word)
                self.index_2_word.append(k)
            else:
                self.word_2_index[k] = 0
        self.en_set = set()
        self.en_count = {}
        p = 0
        for data in self.zh_data:
            if p % 1000 == 0:
                print('中文', p / le)
            sentence = list(jieba.cut(data))
            for sen in sentence:
                if sen in self.en_set:
                    self.en_count[sen] += 1
                else:
                    self.en_set.add(sen)
                    self.en_count[sen] = 1
            p += 1
        for k, v in self.en_count.items():
            if v >= self.__count_min:
                self.word_2_index[k] = len(self.index_2_word)
                self.index_2_word.append(k)
            else:
                self.word_2_index[k] = 0

        # 打印词汇表大小和前几个词汇
        print(f'词汇表大小: {len(self.index_2_word)}')
        print('前10个词汇:', self.index_2_word[:10])

    def __filter_data(self):
        length = len(self.en_data)
        self.data_ = []
        for i in range(length):
            self.data_.append([self.en_data[i], self.zh_data[i], 0])
            self.data_.append([self.zh_data[i], self.en_data[i], 1])

    def en_cut(self, data):
        data = word_tokenize(data)
        if len(data) > 40:
            return 0, []
        en_tokens = []
        for tk in data:
            en_tokens.append(self.word_2_index.get(tk, 0))
        return 1, en_tokens

    def zh_cut(self, data):
        data = list(jieba.cut(data))
        if len(data) > 40:
            return 0, []
        en_tokens = []
        for tk in data:
            en_tokens.append(self.word_2_index.get(tk, 0))
        return 1, en_tokens

    def encode_all(self, data):
        src = []
        tgt = []
        en_src, en_tgt, l = [], [], []
        labels = []
        for i in data:
            en_src.append(i[0])
            en_tgt.append(i[1])
            l.append(i[2])
        for i in range(len(l)):
            if l[i] == 0:
                lab1, src_tokens = self.en_cut(en_src[i])
                if lab1 == 0:
                    continue
                lab2, tgt_tokens = self.zh_cut(en_tgt[i])
                if lab2 == 0:
                    continue
                src.append(src_tokens)
                tgt.append(tgt_tokens)
                labels.append(i)
            else:
                lab1, tgt_tokens = self.en_cut(en_tgt[i])
                if lab1 == 0:
                    continue
                lab2, src_tokens = self.zh_cut(en_src[i])
                if lab2 == 0:
                    continue
                src.append(src_tokens)
                tgt.append(tgt_tokens)
                labels.append(i)
        return labels, src, tgt

    def encode(self, src, l):
        if l == 0:
            src1 = word_tokenize(src)
            en_tokens = []
            for tk in src1:
                en_tokens.append(self.word_2_index.get(tk, 0))
            return [en_tokens]
        else:
            src1 = list(jieba.cut(src))
            en_tokens = []
            for tk in src1:
                en_tokens.append(self.word_2_index.get(tk, 0))
            return [en_tokens]

    def decode(self, data):
        """
        数据解码
        :param data: 这里传入一个句子（单词索引列表）
        :return: 返回解码后的句子
        """
        return ' '.join([self.index_2_word[idx] for idx in data if idx < len(self.index_2_word)])

    def __get_datasets(self, data):
        labels, src, tgt = self.encode_all(data)
        return TranslationDataset(src, tgt)

    def another_process(self, batch_datas):
        en_index, zh_index = [], []
        en_len, zh_len = [], []

        for en, zh in batch_datas:
            en_index.append(en)
            zh_index.append(zh)
            en_len.append(len(en))
            zh_len.append(len(zh))

        max_en_len = max(en_len)
        max_zh_len = max(zh_len)
        max_len = max(max_en_len, max_zh_len + 2)

        en_index = [i + [self.word_2_index['<pad>']] * (max_len - len(i)) for i in en_index]
        zh_index = [[self.word_2_index['<bos>']] + i + [self.word_2_index['<eos>']] +
                    [self.word_2_index['<pad>']] * (max_len - len(i) + 1) for i in zh_index]

        en_index = torch.tensor(en_index)
        zh_index = torch.tensor(zh_index)
        return en_index, zh_index

    def get_dataloader(self, data, batch_size=64):  # 修改批次大小为64
        data = self.__get_datasets(data)
        return DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=self.another_process)

    def get_vocab_size(self):
        return len(self.index_2_word)

    def get_dataset_size(self):
        return len(self.en_data)

# 定义Batch类
class Batch:
    def __init__(self, src, trg=None, tokenizer=None, device='cuda'):
        src = src.to(device).long()
        trg = trg.to(device).long()
        self.src = src
        self.__pad = tokenizer.word_2_index['<pad>']
        self.src_mask = (src != self.__pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, self.__pad)
            self.ntokens = (self.trg_y != self.__pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(Batch.subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

    @staticmethod
    def subsequent_mask(size):
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        return torch.from_numpy(subsequent_mask) == 0

# 定义Embedding类
class Embedding(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embedding, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

# 定义PositionalEncoding类
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.2, max_len=5000, device='cuda'):  # 修改dropout比例为0.2
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0.0, max_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, d_model, 2, device=device) * (-math.log(1e4) / d_model)).unsqueeze(0)
        pe[:, 0::2] = torch.sin(torch.mm(position, div_term))
        pe[:, 1::2] = torch.cos(torch.mm(position, div_term))
        pe.unsqueeze_(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x += Variable(self.pe[:, :x.size(1), :], requires_grad=False)
        return self.dropout(x)

# 定义MultiHeadedAttention类
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.2):  # 修改dropout比例为0.2
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([copy.deepcopy(nn.Linear(d_model, d_model)) for _ in range(4)])
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]
        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    @staticmethod
    def attention(query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

# 定义SublayerConnection类
class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        x_ = self.norm(x)
        x_ = sublayer(x_)
        x_ = self.dropout(x_)
        return x + x_

# 定义FeedForward类
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.2):  # 修改dropout比例为0.2
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.w_2(x)
        return x

# 定义Encoder类
class Encoder(nn.Module):
    def __init__(self, h, d_model, d_ff=256, dropout=0.2):  # 修改dropout比例为0.2
        super(Encoder, self).__init__()
        self.self_attn = MultiHeadedAttention(h, d_model, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, mask))
        return self.norm(self.sublayer2(x, self.feed_forward))

# 定义Decoder类
class Decoder(nn.Module):
    def __init__(self, h, d_model, d_ff=256, dropout=0.2):  # 修改dropout比例为0.2
        super(Decoder, self).__init__()
        self.size = d_model
        self.self_attn = MultiHeadedAttention(h, d_model, dropout)
        self.src_attn = MultiHeadedAttention(h, d_model, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)
        self.sublayer3 = SublayerConnection(d_model, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer2(x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.norm(self.sublayer3(x, self.feed_forward))

# 定义Generator类
class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

# 定义Transformer类
class Transformer(nn.Module):
    def __init__(self, tokenizer, h=8, d_model=256, E_N=2, D_N=2, device='cuda'):
        super(Transformer, self).__init__()
        self.encoder = nn.ModuleList([Encoder(h, d_model) for _ in range(E_N)])
        self.decoder = nn.ModuleList([Decoder(h, d_model) for _ in range(D_N)])
        self.src_embed = Embedding(d_model, tokenizer.get_vocab_size())
        self.tgt_embed = Embedding(d_model, tokenizer.get_vocab_size())
        self.src_pos = PositionalEncoding(d_model, device=device)
        self.tgt_pos = PositionalEncoding(d_model, device=device)
        self.generator = Generator(d_model, tokenizer.get_vocab_size())

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        for i in self.encoder:
            src = i(src, src_mask)
        return src

    def decode(self, memory, tgt, src_mask, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        for i in self.decoder:
            tgt = i(tgt, memory, src_mask, tgt_mask)
        return tgt

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), tgt, src_mask, tgt_mask)

# 定义LabelSmoothing类
class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

# 定义SimpleLossCompute类
class SimpleLossCompute:
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data.item() * norm.float()

# 定义NoamOpt类
class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        self.optimizer.zero_grad()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

# 定义is_english_sentence函数
def is_english_sentence(sentence):
    english_pattern = re.compile(r'[a-zA-Z]')
    match = english_pattern.search(sentence)
    return bool(match)

# 定义compute_bleu4函数
def compute_bleu4(tokenizer, random_integers, model, device):
    m1, m2 = [], []
    m3, m4 = [], []
    model.eval()
    da = []
    for i in random_integers:
        da.append(tokenizer.test[i])
    labels, x, _ = tokenizer.encode_all(da)
    with torch.no_grad():
        y = predict(x, model, tokenizer, device)
    p = 0
    itg = []
    if len(y) != 10:
        return 0
    for i in labels:
        itg.append(random_integers[i])
    for i in itg:
        if is_english_sentence(tokenizer.test[i][1]):
            m1.append(tokenizer.test[i][1])
            m2.append([y[p]])
        else:
            m3.append(list(jieba.cut(tokenizer.test[i][1])))
            m4.append([list(jieba.cut(y[p]))])
        p += 1
    smooth = SmoothingFunction().method1
    b1 = [sacrebleu.sentence_bleu(candidate, refs).score for candidate, refs in zip(m1, m2)]
    for i in range(len(m4)):
        b2 = sentence_bleu(m4[i], m3[i], weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth) * 100
        b1.append(b2)
    return sum(b1) / len(b1)

# 定义validate函数
def validate(tokenizer, model, device='cuda'):
    model.eval()  # 设置模型为评估模式
    random_integers = random.sample(range(len(tokenizer.test)), 10)  # 随机选择一些测试集数据进行验证
    bleu4_score = compute_bleu4(tokenizer, random_integers, model, device)
    print(f"Validation BLEU4 score: {bleu4_score:.2f}")
    return bleu4_score  # 返回BLEU4分数

# 定义greedy_decode函数
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(memory, ys, src_mask, Variable(Batch.subsequent_mask(ys.size(1)).type_as(src.data)))
        prob = model.generator(out[:, i])
        _, next_word = torch.max(prob, dim=-1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

# 定义predict函数
def predict(data, model, tokenizer, device='cuda'):
    with torch.no_grad():
        data1 = []
        for i in range(len(data)):
            src = torch.from_numpy(np.array(data[i])).long().to(device)
            src = src.unsqueeze(0)
            src_mask = (src != tokenizer.word_2_index['<pad>']).unsqueeze(-2)
            out = greedy_decode(model, src, src_mask, max_len=100, start_symbol=tokenizer.word_2_index['<bos>'])
            translation = []
            for j in range(1, out.size(1)):
                sym = tokenizer.index_2_word[out[0, j].item()]
                if sym != '<eos>':
                    translation.append(sym)
                else:
                    break
            if len(translation) > 0:
                if translation[0].lower() in words.words():
                    translated_sentence = TreebankWordDetokenizer().detokenize(translation)
                else:
                    translated_sentence = "".join(translation)
                data1.append(translated_sentence)
                print(f'原句: {tokenizer.decode(data[i])}')
                print(f'Predicted: {translated_sentence}')
                print(f'Translation tokens: {translation}')
                print(f'Source tokens: {data[i]}')
        return data1


# 定义train函数
def train():
    device = 'cuda'
    model = Transformer(tokenizer, device=device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    model = model.to(device)
    criteria = LabelSmoothing(tokenizer.get_vocab_size(), tokenizer.word_2_index['<pad>'])
    optimizer = NoamOpt(256, 1, 4000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))  # 调整warmup到4000
    lossF = SimpleLossCompute(model.generator, criteria, optimizer)
    epochs = 100
    model.train()
    loss_all = []
    bleu4_scores = []  # 用于记录每个epoch的BLEU4分数
    print('词表大小', tokenizer.get_vocab_size())
    data_loader = tokenizer.get_dataloader(tokenizer.data_)
    random_integers = random.sample(range(len(tokenizer.test) - 10), 6)
    batchs = []
    for index, data in enumerate(data_loader):
        src, tgt = data
        batch = Batch(src, tgt, tokenizer=tokenizer, device=device)
        batchs.append(batch)
    for epoch in range(epochs):
        epoch_start_time = time.time()
        print(f'Starting epoch {epoch} at {time.ctime(epoch_start_time)}')
        p = 0
        for batch in batchs:
            out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
            loss = lossF(out, batch.trg_y, batch.ntokens)
            if (p + 1) % 1000 == 0:
                model.eval()
                print(f'Epoch {epoch}, Batch {p}, Loss {loss.item() / batch.ntokens}, Time {time.time() - epoch_start_time}')
                model.train()
            if p % 100 == 0:
                print(f'Processing batch {p} of epoch {epoch}')
            p += 1
        loss_all.append(float(loss.item() / batch.ntokens))
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f'Epoch {epoch} finished at {time.ctime(epoch_end_time)} with loss {float(loss.item() / batch.ntokens)}. Duration: {epoch_duration:.2f} seconds')

        # 在每个epoch结束后进行验证
        bleu4_score = validate(tokenizer, model, device)
        bleu4_scores.append(bleu4_score)  # 记录BLEU4分数
        if bleu4_score > 14:
            model_path = f'model/transformer_epoch_{epoch}_bleu4_{bleu4_score:.2f}.pt'
            torch.save(model.state_dict(), model_path)
            print(f'Model saved to {model_path}')

    with open('loss.txt', 'w', encoding='utf-8') as f:
        f.write(str(loss_all))

    # 绘制loss变化曲线
    plt.figure()
    plt.plot(range(epochs), loss_all, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.savefig('loss_curve.png')  # 保存loss变化曲线图
    plt.show()

    # 绘制BLEU4分数变化曲线
    plt.figure()
    plt.plot(range(epochs), bleu4_scores, label='Validation BLEU4 Score')
    plt.xlabel('Epoch')
    plt.ylabel('BLEU4 Score')
    plt.title('BLEU4 Score over Epochs')
    plt.legend()
    plt.savefig('bleu4_curve.png')  # 保存BLEU4分数变化曲线图
    plt.show()

if __name__ == '__main__':
    en_path = r'.\\dataset\\train_en.txt'
    zh_path = r'.\\dataset\\train_zh.txt'
    tokenizer = Tokenizer(en_path, zh_path, count_min=3)

    # 确保model文件夹存在
    if not os.path.exists('model'):
        os.makedirs('model')

    # 开始训练
    train()
