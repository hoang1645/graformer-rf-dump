import sentencepiece as spm
import torch
from transformers.tokenization_utils_base import BatchEncoding



# Tried to mimic as close to HuggingFace's PreTrainedTokenizers as possible
class SentencePieceTokenizer(object):
    def __init__(self, path):
        self.sp = spm.SentencePieceProcessor(model_file=path)
        self.pad_token_id = self.sp.pad_id()
        self.pad_token = self.sp.id_to_piece(self.pad_token_id)
        
        self.bos_token_id = self.sp.bos_id()
        self.bos_token = self.sp.id_to_piece(self.bos_token_id)

        self.eos_token_id = self.sp.eos_id()
        self.eos_token = self.sp.id_to_piece(self.eos_token_id)
        
        self.unk_token_id = self.sp.unk_id()
        self.unk_token = self.sp.id_to_piece(self.unk_token_id)

        self.vocab_size = self.sp.vocab_size()
    
    def tokenize(self, text):
        return self.sp.tokenize(text, out_type=str)
    
    def convert_tokens_to_ids(self, tokens:list[list[str]], return_tensors='pt'):
        ids = []
        for token in tokens:
            ids.append(self.sp.PieceToId(token))
        if return_tensors == 'pt': ids = torch.LongTensor(ids)
        return ids
    def __call__(self, texts, return_tensors='pt', padding=True, pad_left=True):
        ids = self.sp.tokenize(texts)
        if return_tensors == 'pt': 
            max_len = 0
            for id_ in ids:
                if len(id_) > max_len: max_len = len(id_)
            t_ids = torch.ones(size=(len(ids), max_len + 2), dtype=torch.int, )
            attention_mask = torch.zeros_like(t_ids)
            t_ids = t_ids * self.pad_token_id
            t_ids[:, 0] = self.bos_token_id
            for i, id_ in enumerate(ids):
                if pad_left:
                    t_ids[i, 1:len(id_)+1] = torch.Tensor(id_)
                    attention_mask[i, :len(id_)+2] = 1
                else:
                    t_ids[i, -len(id_)-1:-1] = torch.Tensor(id_)
                    attention_mask[i, -len(id_)-2:] = 1
            ids = t_ids.long()
            attention_mask = attention_mask.long()
        else:
            attention_mask = [[1] * len(id_) for id_ in ids]
        return_results = BatchEncoding()
        return_results['input_ids'] = ids
        return_results['attention_mask'] = attention_mask
        return return_results
    
    def batch_decode(self, ids, **kwargs):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self.sp.Decode(ids)