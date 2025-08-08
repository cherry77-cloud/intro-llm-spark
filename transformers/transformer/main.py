import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from models import Transformer
from batch import create_masks
from process import *
import numpy as np
import time
import os


# ---------------------- Data paths ----------------------
BASE_DIR = os.path.dirname(__file__)
src_file = os.path.join(BASE_DIR, 'data', 'english.txt')
trg_file = os.path.join(BASE_DIR, 'data', 'french.txt')

src_lang = 'en_core_web_sm'
trg_lang = 'fr_core_news_sm'
max_strlen = 80
batchsize = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------- Data ----------------------
src_data, trg_data = read_data(src_file, trg_file)
EN_TEXT, FR_TEXT = create_fields(src_lang, trg_lang)
train_iter, src_pad, trg_pad = create_dataset(
    src_data, trg_data, EN_TEXT, FR_TEXT, max_strlen, batchsize)


# ---------------------- Model ----------------------
d_model = 512
heads = 8
N = 6
dropout = 0.1
src_vocab = len(EN_TEXT.vocab)
trg_vocab = len(FR_TEXT.vocab)
model = Transformer(src_vocab, trg_vocab, d_model, N, heads, dropout).to(device)

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)


# ---------------------- Training ----------------------
def train_model(epochs, print_every=100):
    model.train()
    start = time.time()
    temp = start
    total_loss = 0

    for epoch in range(epochs):
        for i, batch in enumerate(train_iter):
            src = batch.src.transpose(0, 1).to(device)
            trg = batch.trg.transpose(0, 1).to(device)

            trg_input = trg[:, :-1]
            targets = trg[:, 1:].contiguous().view(-1)

            src_mask, trg_mask = create_masks(src, trg_input, src_pad, trg_pad)
            if src_mask is not None:
                src_mask = src_mask.to(device)
            if trg_mask is not None:
                trg_mask = trg_mask.to(device)

            preds = model(src, trg_input, src_mask, trg_mask)

            optim.zero_grad()
            loss = F.cross_entropy(
                preds.view(-1, preds.size(-1)), targets, ignore_index=trg_pad)
            loss.backward()
            optim.step()

            total_loss += loss.item()
            if (i + 1) % print_every == 0:
                loss_avg = total_loss / print_every
                elapsed = time.time() - temp
                print(
                    f"time = {(time.time() - start)//60:.0f}m, epoch {epoch+1}, iter = {i+1}, loss = {loss_avg:.3f}, {elapsed:.0f}s per {print_every} iters")
                total_loss = 0
                temp = time.time()


# ---------------------- Translation ----------------------
def translate(src_sentence, max_len=80):
    model.eval()
    src = tokenize_en(src_sentence, EN_TEXT)
    src = torch.LongTensor(src).to(device)

    src_mask = (src != src_pad).unsqueeze(-2)
    e_outputs = model.encoder(src.unsqueeze(0), src_mask)

    outputs = torch.zeros(max_len, device=device, dtype=src.dtype)
    outputs[0] = torch.LongTensor([FR_TEXT.vocab.stoi['<sos>']]).to(device)

    for i in range(1, max_len):
        trg_mask = np.triu(np.ones((1, i, i)).astype('uint8'))
        trg_mask = Variable(torch.from_numpy(trg_mask).to(device) == 0)

        out = model.out(model.decoder(
            outputs[:i].unsqueeze(0), e_outputs, src_mask, trg_mask))
        out = F.softmax(out, dim=-1)
        _, ix = out[:, -1].data.topk(1)
        outputs[i] = ix[0][0]
        if ix[0][0] == FR_TEXT.vocab.stoi['<eos>']:
            break

    return ' '.join([FR_TEXT.vocab.itos[ix] for ix in outputs[:i]])


# ---------------------- Main ----------------------
if __name__ == "__main__":
    train_model(10)
    test_sentence = 'Let me see.'
    print(translate(test_sentence))
