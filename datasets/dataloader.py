from torch.utils.data import DataLoader
from datasets.IWSLTDataset import IWSLTDataset

def get_dataloader(source, target, src_tokenizer, tgt_tokenizer, batch_size, test=False):
    # Both tokenizers should have a __call__ method which is equiv. to `convert_tokens_to_ids(tokenize(x))`
    def collate_fn(batch):
        # print(batch)
        source, target = [b[0] for b in batch], [b[1] for b in batch]
        
        return src_tokenizer(source, return_tensors='pt'), tgt_tokenizer(target, return_tensors='pt')
    return DataLoader(
        dataset=IWSLTDataset(source, target),
        batch_size=batch_size,
        shuffle=not test,
        collate_fn=collate_fn if not test else None
    )
