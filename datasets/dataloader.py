from torch.utils.data import DataLoader
from datasets.IWSLTDataset import IWSLTDataset


def get_dataloader(source, target, src_tokenizer, tgt_tokenizer, batch_size, num_workers=1, test=False):
    # Both tokenizers should have a __call__ method which is equiv. to `convert_tokens_to_ids(tokenize(x))`
    def collate_fn(batch):
        # print(batch)
        _source, _target = [b[0] for b in batch], [b[1] for b in batch]
        _source, _target = src_tokenizer(_source, return_tensors='pt', padding=True), tgt_tokenizer(_target, return_tensors='pt', padding=True)
        return _source, _target
    return DataLoader(
        dataset=IWSLTDataset(source, target),
        batch_size=batch_size,
        shuffle=not test,
        collate_fn=collate_fn if not test else None,
        num_workers=num_workers
    )
