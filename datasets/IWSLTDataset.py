import torch
from torch.utils.data import Dataset


class IWSLTDataset(Dataset):

    def __init__(self, src_fpath, tgt_fpath):
        self.src_sentences = self._read_txt(src_fpath)
        self.tgt_sentences = self._read_txt(tgt_fpath)

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        return self.src_sentences[idx], self.tgt_sentences[idx]

    @staticmethod
    def _read_txt(fpath):
        sentences = list()
        with open(fpath, "r") as f:
            for line in f:
                sentences.append(line.rstrip())
        return sentences