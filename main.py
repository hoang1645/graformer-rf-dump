from models.lightning_model import LightningGraformer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from args.parser import GraformerArgumentParser
from sacrebleu import corpus_bleu
from tqdm import tqdm
from datasets.dataloader import get_dataloader
from transformers import AutoTokenizer
import torch

import os

def main():
    parser = GraformerArgumentParser()
    args = parser.get_args()

    model = LightningGraformer(
            args.masked_encoder, args.causal_decoder, args.d_model, args.n_heads, args.dff, lr=args.lr
        )
    # model.half()
    if not args.test_only:
        # dataloader goes here
        train_dataloader = get_dataloader(args.train_path_src, args.train_path_tgt, 
                                          AutoTokenizer.from_pretrained(args.masked_encoder), 
                                          AutoTokenizer.from_pretrained(args.causal_decoder),
                                          batch_size=args.batch_size)
        val_dataloader = get_dataloader(args.valid_path_src, args.valid_path_tgt, 
                                          AutoTokenizer.from_pretrained(args.masked_encoder), 
                                          AutoTokenizer.from_pretrained(args.causal_decoder),
                                          batch_size=args.batch_size)
        
        if not os.path.isdir('outputs'): os.mkdir('outputs')
        
        routine_pt_callback = ModelCheckpoint(dirpath='outputs', save_last=True, save_weights_only=True, 
                                   every_n_epochs=1, auto_insert_metric_name=True, verbose=True)
        routine_pt_callback.FILE_EXTENSION = '.pt'


        best_pt_callback = ModelCheckpoint(save_top_k=1, save_weights_only=True, 
                                           filename='outputs/best.pt', monitor='val_loss', verbose=True)
        trainer = pl.Trainer(accelerator='gpu', max_epochs=args.epoch, callbacks=[routine_pt_callback, best_pt_callback], 
                             auto_scale_batch_size=True)
        trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=args.from_checkpoint)

    else:
        model.load_from_checkpoint(args.from_checkpoint)
    
    # test
    test_dataloader = val_dataloader = get_dataloader(args.test_path_src, args.test_path_tgt, 
                                          AutoTokenizer.from_pretrained(args.masked_encoder), 
                                          AutoTokenizer.from_pretrained(args.causal_decoder),
                                          batch_size=args.batch_size, test=True) # don't tokenize in test dataloader
    core = model.graformer
    
    translations = []
    targets = []

    for src, tgt in tqdm(test_dataloader):
        trl = core.translate(src)
        translations.extend(trl)
        targets.extend(tgt)
    
    print(corpus_bleu(translations, [targets]))

main()