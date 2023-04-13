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
        
        routine_pt_callback = ModelCheckpoint(dirpath='outputs', save_last=True, save_weights_only=True, 
                                   every_n_epochs=1, auto_insert_metric_name=True, verbose=True)
        routine_pt_callback.FILE_EXTENSION = '.pt'


        best_pt_callback = ModelCheckpoint(save_top_k=1, save_weights_only=True, 
                                           filename='outputs/best.pt', monitor='val_loss', verbose=True)
        trainer = pl.Trainer(accelerator='gpu', max_epochs=args.epoch, callbacks=[routine_pt_callback, best_pt_callback])
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