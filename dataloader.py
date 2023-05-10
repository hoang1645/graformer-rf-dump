from torch.utils.data import DataLoader
from IWSLTDataset import IWSLTDataset
from transformers import PreTrainedTokenizer


def get_dataloader(source, target, src_tokenizer:PreTrainedTokenizer, tgt_tokenizer:PreTrainedTokenizer, 
                   batch_size, num_workers=1, test=False):
    # Both tokenizers should have a __call__ method which is equiv. to `convert_tokens_to_ids(tokenize(x))`
    def collate_fn(batch):
        # print(batch)
        _source, _target = [b[0] for b in batch], [b[1] for b in batch]
        _source, _target = src_tokenizer(_source, return_tensors='pt', padding=True, truncation='longest_first'), \
                        tgt_tokenizer(_target, return_tensors='pt', padding=True, truncation='longest_first')
        return _source, _target
    def test_collate_fn(batch): return [b[0] for b in batch], [b[1] for b in batch]
    return DataLoader(
        dataset=IWSLTDataset(source, target),
        batch_size=batch_size,
        shuffle=not test,
        collate_fn=collate_fn if not test else test_collate_fn,
        # num_workers=num_workers
    )
