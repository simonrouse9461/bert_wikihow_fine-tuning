import torch
import pandas as pd
import numpy as np
from torch.utils.data import IterableDataset
from torch.nn.utils.rnn import pad_sequence


class WikihowNSP(IterableDataset):
 
    def __init__(self, file, shuffle=False, tqdm=None):
        self.data = pd.read_csv(file)
        self.shuffle = shuffle
        self.tqdm = tqdm
        
    def __iter__(self):
        paragraph_idx = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(paragraph_idx)
        if self.tqdm is not None:
            paragraph_idx = self.tqdm(paragraph_idx)
        for p in paragraph_idx:
            raw = self.data['text'][p]
            if type(raw) is not str:
                continue
            sentences = raw.split(". ")
            if len(sentences) <= 2:
                continue
            sentence_idx = np.arange(len(sentences) - 1)
            if self.shuffle:
                np.random.shuffle(sentence_idx)
            for i in sentence_idx:
                rand = np.random.choice(len(sentences) - 2)
                if rand >= i:
                    rand += 2
                yield sentences[i], sentences[i + 1], sentences[rand]

    def loader(self, collator, batch_size=8):
        return torch.utils.data.DataLoader(dataset=self,
                                           batch_size=batch_size // 2,
                                           collate_fn=collator)
    
    def sample(self, collator, batch_size=2):
        return next(iter(self.loader(collator, batch_size=batch_size)))
    
    
class NSPBatchCollator:
    
    def __init__(self, tokenizer, max_len=256, device='cpu'):
        self.device = device
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def to(self, device):
        self.device = device
        return self
    
    def _convert_to_tensor(self, sequence):
        return pad_sequence(list(map(torch.tensor, sequence)), batch_first=True)
    
    def __call__(self, batch):
        sentences, true_next, false_next = zip(*batch)
        pairs = list(zip(sentences * 2, true_next + false_next))
        next_sentence_label = torch.cat([torch.zeros(len(sentences)), 
                                         torch.ones(len(sentences))]).long()
        encoded = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=pairs,
                                                   max_length=self.max_len,
                                                   add_special_tokens=True,
                                                   return_token_type_ids=True,
                                                   return_attention_mask=True)
        input_ids = self._convert_to_tensor(encoded['input_ids'])
        token_type_ids = self._convert_to_tensor(encoded['token_type_ids'])
        attention_mask = self._convert_to_tensor(encoded['attention_mask'])
        return (input_ids.to(self.device), 
                token_type_ids.to(self.device), 
                attention_mask.to(self.device), 
                next_sentence_label.to(self.device))
