import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import tokenize
from collections import Counter
from pathlib import Path


# ---------------------- Vocab & Field ----------------------
class Vocab:
    def __init__(self, stoi):
        self.stoi = stoi
        self.itos = [None] * len(stoi)
        for tok, idx in stoi.items():
            self.itos[idx] = tok

    def __len__(self):
        return len(self.stoi)


class Field:
    def __init__(self, tokenize_fn, lower=True, init_token=None, eos_token=None, pad_token="<pad>"):
        self.tokenize = tokenize_fn
        self.lower = lower
        self.init_token = init_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.vocab = None

    def build_vocab(self, sentences):
        counter = Counter()
        for s in sentences:
            tokens = self.tokenize(s)
            if self.lower:
                tokens = [t.lower() for t in tokens]
            if self.init_token:
                tokens = [self.init_token] + tokens
            if self.eos_token:
                tokens.append(self.eos_token)
            counter.update(tokens)

        specials = [self.pad_token, '<unk>']
        if self.init_token and self.init_token not in specials:
            specials.append(self.init_token)
        if self.eos_token and self.eos_token not in specials:
            specials.append(self.eos_token)

        stoi = {tok: idx for idx, tok in enumerate(specials)}
        for tok, _ in counter.most_common():
            if tok not in stoi:
                stoi[tok] = len(stoi)

        self.vocab = Vocab(stoi)

    def numericalize(self, sentence):
        tokens = self.tokenize(sentence)
        if self.lower:
            tokens = [t.lower() for t in tokens]
        if self.init_token:
            tokens = [self.init_token] + tokens
        if self.eos_token:
            tokens.append(self.eos_token)
        return [self.vocab.stoi.get(t, self.vocab.stoi['<unk>']) for t in tokens]


# ---------------------- Dataset ----------------------
class TranslationDataset(Dataset):
    def __init__(self, src_sentences, trg_sentences, src_field: Field, trg_field: Field):
        self.src_field = src_field
        self.trg_field = trg_field
        self.src_data = [src_field.numericalize(s) for s in src_sentences]
        self.trg_data = [trg_field.numericalize(s) for s in trg_sentences]

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return torch.tensor(self.src_data[idx], dtype=torch.long), torch.tensor(self.trg_data[idx], dtype=torch.long)


def pad_sequences(sequences, pad_value):
    max_len = max([seq.size(0) for seq in sequences])
    padded = torch.full((max_len, len(sequences)), pad_value, dtype=torch.long)
    for i, seq in enumerate(sequences):
        padded[:seq.size(0), i] = seq
    return padded


def collate_fn(pad_idx_src, pad_idx_trg):
    def _collate(batch):
        src_seqs, trg_seqs = zip(*batch)
        src_batch = pad_sequences(src_seqs, pad_idx_src)  # (seq_len, batch)
        trg_batch = pad_sequences(trg_seqs, pad_idx_trg)  # (seq_len, batch)
        return type('Batch', (), {'src': src_batch, 'trg': trg_batch})
    return _collate


# ---------------------- Public API (replacing torchtext) ----------------------
def read_data(src_file, trg_file):
    if not Path(src_file).exists():
        raise FileNotFoundError(f"Source file '{src_file}' not found")
    if not Path(trg_file).exists():
        raise FileNotFoundError(f"Target file '{trg_file}' not found")

    with open(src_file, encoding='utf-8') as f:
        src_data = [line.strip() for line in f if line.strip()]
    with open(trg_file, encoding='utf-8') as f:
        trg_data = [line.strip() for line in f if line.strip()]
    return src_data, trg_data


def create_fields(src_lang, trg_lang):
    t_src = tokenize(src_lang)
    t_trg = tokenize(trg_lang)
    SRC = Field(t_src.tokenizer, lower=True)
    TRG = Field(t_trg.tokenizer, lower=True,
                init_token='<sos>', eos_token='<eos>')
    return SRC, TRG


def create_dataset(src_data, trg_data, SRC: Field, TRG: Field, max_strlen, batchsize, build_vocab=True):
    """Create DataLoader. If build_vocab=False, assume SRC/TRG vocab already built."""
    filtered_src = []
    filtered_trg = []
    for s, t in zip(src_data, trg_data):
        if s.count(' ') < max_strlen and t.count(' ') < max_strlen:
            filtered_src.append(s)
            filtered_trg.append(t)

    # Build vocabulary (only first time)
    if build_vocab or SRC.vocab is None or TRG.vocab is None:
        SRC.build_vocab(filtered_src)
        TRG.build_vocab(filtered_trg)

    src_pad = SRC.vocab.stoi['<pad>']
    trg_pad = TRG.vocab.stoi['<pad>']

    dataset = TranslationDataset(filtered_src, filtered_trg, SRC, TRG)
    loader = DataLoader(dataset, batch_size=batchsize, shuffle=True, collate_fn=collate_fn(
        src_pad, trg_pad), num_workers=0, pin_memory=True)

    return loader, src_pad, trg_pad


# ---------------------- Utility for translation ----------------------
_spacy_en_tok = None

def tokenize_en(sentence, SRC: Field):
    global _spacy_en_tok
    if _spacy_en_tok is None:
        _spacy_en_tok = tokenize('en_core_web_sm')
    tokens = _spacy_en_tok.tokenizer(sentence)
    tokens = [tok.lower() for tok in tokens]
    if SRC.init_token:
        tokens = [SRC.init_token] + tokens
    if SRC.eos_token:
        tokens.append(SRC.eos_token)
    return [SRC.vocab.stoi.get(tok, SRC.vocab.stoi['<unk>']) for tok in tokens]
