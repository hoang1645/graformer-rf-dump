from models.lightning_model import LightningGraformer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from args.parser import GraformerArgumentParser
import nltk
from tqdm import tqdm
import os

def main():
    parser = GraformerArgumentParser()
    args = parser.get_args()

    model = LightningGraformer(
            args.masked_encoder, args.causal_decoder, args.d_model, args.n_heads, args.dff, lr=args.lr
        )
    
    if not args.test_only:
        # dataloader goes here
        train_dataloader = ...
        val_dataloader = ...

        if not os.path.isdir('outputs'): os.mkdir('outputs')
        
        callback = ModelCheckpoint(save_last=True, save_weights_only=True, every_n_epochs=1)

        trainer = pl.Trainer(accelerator='gpu', max_epochs=args.epoch, callbacks=callback)
        trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=args.from_checkpoint)

    else:
        model.load_from_checkpoint(args.from_checkpoint)
    
    # test
    test_dataloader = ... # don't tokenize in test dataloader
    core = model.graformer
    bleu = 0
    for src, tgt in tqdm(test_dataloader):
        trl = core.translate(src)
        for trl_, tgt_ in zip(trl, tgt):
            trl_ = trl_.split()
            tgt_ = tgt.split()
            bleu += nltk.bleu([tgt_], trl_)
    
    print(bleu)