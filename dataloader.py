import os
import numpy as np
import torch
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def file2list(label_file):
    def line_proc(line):
        line = line.strip('\n').split('|')
        return [line[0], line[2]]

    with open(label_file, 'r') as fr:
        lines = fr.readlines()
    labels = list(map(line_proc, lines))
    return labels

class MelDataset(torch.utils.data.Dataset):
    def __init__(self, wav_root, label_path, extension=".npy"):
        super(MelDataset, self).__init__()
        self.wav_root = wav_root
        self.dataset = file2list(label_path)
        self.extension = extension
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        mel_name, text = self.dataset[i] 
        mel_name = mel_name.split('.')[0]
        mel = np.load(os.path.join(self.wav_root, mel_name + self.extension)) 
        ids = tokenizer.encode(text)
        return mel, ids, text

def _collate_fn(batch):
    max_seq_size = max([s[0].shape[1] for s in batch])
    max_label_size = max([len(s[1]) for s in batch])
    feat_size = batch[0][0].shape[0]
    seqs = torch.zeros(len(batch), feat_size, max_seq_size)
    labels = torch.zeros(len(batch), max_label_size, dtype=torch.int64)
    text_bin = []
    for i in range(len(batch)):
        text_bin.append(batch[i][2])
        
        seq = batch[i][0]
        seq_length = seq.shape[1]
        seqs[i].narrow(1, 0, seq_length).copy_(torch.from_numpy(seq))
        
        label = batch[i][1]
        label_length = len(label)
        labels[i].narrow(0, 0, label_length).copy_(torch.tensor(label))
    return seqs, labels, tuple(text_bin)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    label_path = "./data/LJSpeech/sample.csv"
    wav_path = "./data/LJSpeech/mels_sample/"

    dset = MelDataset(wav_path, label_path)
    print(tokenizer.pad_token_id)
    
    loader = DataLoader(dset, batch_size=4, collate_fn=_collate_fn)
    
    a = next(iter(loader))
    print(a[0].shape)
