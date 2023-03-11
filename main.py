import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import json
from torch.nn.utils.rnn import pad_sequence

PATH = './Transformer_save.pth'

cn_seqcount = json.load(open('./cn_seqcount.json'))
jp_seqcount = json.load(open('./jp_seqcount.json'))
sentence = json.load(open('./sentence.json'))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 100
# 对应语言都要分别建立词表
sentences = sentence
src_vocab = cn_seqcount
src_idx2word = {i: w for i, w in enumerate(src_vocab)}
src_vocab_size = len(src_vocab)

tgt_vocab = jp_seqcount
idx2word = {i: w for i, w in enumerate(tgt_vocab)}
tgt_vocab_size = len(tgt_vocab)

# Transformer parameters
d_model = 512  # 词向量和位置向量的维度
d_ff = 2048  # 前向传播的维度
d_k = d_v = 64  # k和v矩阵的维度（k == q)
n_layer = 6  # 有几个decoder和encoder层
n_heads = 8  # 多头注意力机制的头数


# 数据集构建：把句子构建
def make_data(sentences):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
        enc_input = [[src_vocab[n] for n in sentences[i][0].split()]]
        dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]]
        dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]]

        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)
    enc_inputs = pad_sequence([torch.tensor(dim) for dim in enc_inputs], batch_first=True, padding_value=0)
    dec_inputs = pad_sequence([torch.tensor(dim) for dim in dec_inputs], batch_first=True, padding_value=0)
    dec_outputs = pad_sequence([torch.tensor(dim) for dim in dec_outputs], batch_first=True, padding_value=0)
    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)


enc_inputs, dec_inputs, dec_outputs = make_data(sentences)


class MyDataSet(Data.Dataset):
    """自定义Dataloader"""

    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]


loader = Data.DataLoader(dataset=MyDataSet(enc_inputs, dec_inputs, dec_outputs), batch_size=8)


# Transformer model
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        :param x:[seq_len, batch_size, d_model]
        :return:
        '''
        x = x + self.pe[:x.size(0), :]  # 把输入x直接加上位置讯息
        return self.dropout(x)


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is pad token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size,len_q,len_k] 构成的一个立方体(batch_size个这样的矩阵)


def get_attn_subsequence_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]  # [batch_size, tgt_len, tgt_len]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # 生成一个上三角矩阵
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        '''

        :param Q:[batch_size, n_heads, len_q, d_k]
        :param K: [batch_size, n_heads, len_k, d_k]
        :param V: [batch_size, n_heads, len_v, d_v]
        :param attn_mask: [batch_size, n_heads, seq_len, seq_len]
        :return:
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # [batch_size, n_heads, len_q ,len_k]
        # mask矩阵填充scores（用-1e9填充scores中与attn_mask中值为1位置相对应的元素）
        scores.masked_fill_(attn_mask, -1e9)

        attn = nn.Softmax(dim=-1)(scores)  # 对最后一个维度(v)做softmax
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        # context :[[z1,z2,...][...]]向量，attn注意力稀疏矩阵，用于可视化
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''

        :param input_Q: [batch_size, len_q,d_model]
        :param input_K: [batch_size, len_k,d_model]
        :param input_V: [batch_size, len_v(=len_k),d_model]
        :param attn_mask: [batch_size, seq_len,seq_len]
        :return:
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # 多头参数矩阵放在一起线性变换，然后再拆成多头
        # Q:[batch_size, n_heads, len_q, d_k]
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)
        # 因为是多头的所以mask矩阵要扩充成四维矩阵
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)  # [batch_size, n_heads, len_q, seq_len]
        context, attn = ScaledDotProductAttention()(Q, K, V,
                                                    attn_mask)  # [batch_size, n_heads, len_q, d_v] [batch_size, n_heads, len_q, len_k]
        # 将不同的头的输出向量拼接在一起
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)  # [batch_size,len_q, n_heads * d_v]
        # 再做一个projection
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model).to(device)(output + residual), attn  # 残差网络（+residual）


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        """
        inputs: [batch_size, seq_len, d_model]
        """
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).to(device)(output + residual)  # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''

        :param enc_inputs: [batch_size, src_len, d_model]
        :param enc_self_attn_mask: [batch_size, src_len, src_len] mask矩阵
        :return:
        :detail: 第一个enc_inputs * W_Q = Q 第二个enc_inputs * W_K = K 第三个enc_inputs * W_V = V
        '''
        # attn [batch_size, n_heads, src_len, src_len]

        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)  # [batch_size, src_len, d_model]
        return enc_outputs, attn


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layer)])  # 把N层encoder所得的矩阵拼接起来方便下面操作

    def forward(self, enc_inputs):
        '''

        :param enc_inputs: [batch_size, sec_len]
        '''
        enc_outputs = self.src_emb(enc_inputs)
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, src_len, d_model]
        # Encoder 输入的p mask(P是不需要进行注意力计算的)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        # enc_outputs:[batch_size, src_len, d_model]
        return enc_outputs, enc_self_attns


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layer)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        '''

        :param dec_inputs: [batch_size, tgt_len]
        :param enc_inputs: [batch_size, src_len]
        :param enc_outputs:[batch_size, src_len, d_model] # 用在Encoder-Decoder Attention层
        :return:
        '''
        dec_outputs = self.tgt_emb(dec_inputs)  # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1).to(
            device)  # [batch_size, tgt_len, d_model]
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).to(device)
        # Masked Self_Attention : 当前时刻是看不到未来的信息的
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).to(device)

        # Decoder 中把两种mask矩阵相加（又屏蔽了pad的信息，也屏蔽了未来时刻的信息）
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0).to(device)

        # 这个mask主要用于E-D attention层
        # get_attn_pad_mask 主要是enc_inputs的pad mask矩阵
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        # dec_outputs:[batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder().to(device)
        self.decoder = Decoder().to(device)
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False).to(device)

    def forward(self, enc_inputs, dec_inputs):
        '''
        :param enc_inputs: [batch_size, sec_len]
        :param dec_inputs: [batch_size, sec_len]
        '''

        enc_outputs, enc_self_attns, = self.encoder(enc_inputs)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs)
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns


model = Transformer().to(device)
# 这里损失函数里面设置了一个参数 ignore_index= 0， 因为“pad“ 这个单词的索引为0，这样设置以后，就不会计算”pad“的损失
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

#训练
for epoch in range(epochs):
    for enc_inputs, dec_inputs, dec_outputs in loader:
        enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        loss = criterion(outputs, dec_outputs.view(-1))  # dec_outputs.view(-1):[batch_size * tgt_len * tgt_vocab_size]
        print('Epoch:', '%04d' % (epoch + 1), 'loss = ', '{:.6f}'.format(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
# save the parameters trained from model
torch.save(model.state_dict(), PATH)


def greedy_decoder(model, enc_input, start_symbol):
    enc_outputs, enc_self_attns = model.encoder(enc_input)
    dec_input = torch.zeros(1, 0).type_as(enc_input.data)
    terminal = False
    next_symbol = start_symbol
    while not terminal:
        dec_input = torch.cat([dec_input.to(device), torch.tensor([[next_symbol]], dtype=enc_input.dtype).to(device)],
                              -1)
        dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[-1]
        next_symbol = next_word
        if next_symbol == tgt_vocab["E"]:
            terminal = True
    greedy_dec_predict = dec_input[:, 1:]
    return greedy_dec_predict


# load the parameters from saved model
model = Transformer()
model.load_state_dict(torch.load(PATH))
# 预测阶段
sentences = [
    # enc_input                dec_input           dec_output
    ['我 能 吃饭 吗 P ', 'S ', ' E'],
    ['我 爱 你 P ', 'S ', ' E'],
    ['和 我 一起 学习 P ', 'S ', ' E']
]

enc_inputs, dec_inputs, dec_outputs = make_data(sentences)
test_loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 3, True)
enc_inputs, _, _ = next(iter(test_loader))

for i in range(len(enc_inputs)):
    greedy_dec_predict = greedy_decoder(model, enc_inputs[i].view(1, -1).to(device), start_symbol=tgt_vocab["S"])
    print(enc_inputs[i], '->', greedy_dec_predict.squeeze())
    print([src_idx2word[t.item()] for t in enc_inputs[i]], '->',
          [idx2word[n.item()] for n in greedy_dec_predict.squeeze()])
