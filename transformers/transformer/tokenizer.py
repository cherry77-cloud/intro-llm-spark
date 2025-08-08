import spacy
import re


class tokenize(object):
    def __init__(self, lang):
        try:
            print(f"Loading spaCy model '{lang}'...", flush=True)
            self.nlp = spacy.load(lang)
        except OSError:
            print(f"Model '{lang}' not found, downloading...", flush=True)
            from spacy.cli import download
            download(lang)
            self.nlp = spacy.load(lang)
        print(f"spaCy model '{lang}' ready.", flush=True)

    def tokenizer(self, sentence):
        sentence = re.sub(r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sentence))
        sentence = re.sub(r"[ ]+", " ", sentence)
        sentence = re.sub(r"\!+", "!", sentence)
        sentence = re.sub(r"\,+", ",", sentence)
        sentence = re.sub(r"\?+", "?", sentence)
        sentence = sentence.lower()
        return [tok.text for tok in self.nlp.tokenizer(sentence) if tok.text != " "]
