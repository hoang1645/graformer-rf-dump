from models.lightning_model import LightningGraformer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import MixedPrecisionPlugin
from args.parser import GraformerArgumentParser
from sacrebleu import corpus_bleu
from tqdm import tqdm
from datasets.dataloader import get_dataloader
import torch
from models.botch import *
from tokenizer.sentencepiece_tokenizer import *
from pytorch_lightning.strategies.ddp import DDPStrategy
import os
from torchinfo import summary


def main():
    parser = GraformerArgumentParser()
    args = parser.get_args()

    if args.tensor_core_precision is not None: torch.set_float32_matmul_precision(args.tensor_core_precision)
    
    # tokenizer = SentencePieceTokenizer('sentencepiece.model')

    # botched_bert, botched_gpt = get_model_with_different_embedding_layer(args.masked_encoder, args.causal_decoder, 
                                                                        #  tokenizer.vocab_size, tokenizer.pad_token_id)

    model = LightningGraformer(
            args.masked_encoder, args.causal_decoder, args.d_model, args.n_heads, args.dff, 
            # encoder_tokenizer=tokenizer, decoder_tokenizer=tokenizer
        )
    summary(model)
    if os.name != 'nt' and args.compile: model = torch.compile(model, backend='inductor', mode='reduce-overhead')
    # model.half()
    if not args.test_only:
        if args.load_only_weights and args.from_checkpoint is not None:
            ckpt_dict = torch.load(args.from_checkpoint)
            model.load_state_dict(ckpt_dict['state_dict'])
            
        # dataloader goes here
        train_dataloader = get_dataloader(args.train_path_src, args.train_path_tgt, 
                                          model.graformer.encoder_tokenizer, 
                                          model.graformer.decoder_tokenizer,
                                          batch_size=args.batch_size, num_workers=args.num_workers)
        val_dataloader = get_dataloader(args.valid_path_src, args.valid_path_tgt, 
                                          model.graformer.encoder_tokenizer, 
                                          model.graformer.decoder_tokenizer,
                                          batch_size=args.batch_size)
        
        # for src, tgt in train_dataloader: print(src.input_ids.shape, tgt.input_ids.shape)
        
        if not os.path.isdir('outputs'): os.mkdir('outputs')
        
        routine_pt_callback = ModelCheckpoint(dirpath='outputs', save_last=True,
                                   every_n_epochs=1, auto_insert_metric_name=True, verbose=True)
        routine_pt_callback.FILE_EXTENSION = '.pt'


        best_pt_callback = ModelCheckpoint(dirpath='outputs', save_top_k=1, save_weights_only=True, 
                                           filename='best', monitor='val_loss', verbose=True)
        best_pt_callback.FILE_EXTENSION = '.pt'
        trainer = pl.Trainer(accelerator='auto', devices=-1,
                             max_epochs=args.epoch, callbacks=[routine_pt_callback, best_pt_callback], 
                             strategy=DDPStrategy(find_unused_parameters=True) if torch.cuda.device_count() > 1 else 'auto',
                             plugins=MixedPrecisionPlugin('16-mixed', 'cuda', torch.cuda.amp.grad_scaler.GradScaler())
                             )
        
        trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=None if args.load_only_weights else args.from_checkpoint)
        # MixedPrecisionPlugin()
    
    else:
        state_dict = torch.load(args.from_checkpoint)['state_dict']
        model.load_state_dict(state_dict)
    
    # test
    test_dataloader = get_dataloader(args.test_path_src, args.test_path_tgt, 
                                          model.graformer.encoder_tokenizer, 
                                          model.graformer.decoder_tokenizer,
                                          batch_size=args.batch_size, test=True) # don't tokenize in test dataloader
    core = model.graformer.half().to('cuda')
    
    translations = []
    targets = []

    for src, tgt in (t:=tqdm(test_dataloader, desc="Evaluating on test dataset")):
        trl = core.translate(src)
        print(trl)
        translations.extend(trl)
        targets.extend(tgt)

        t.set_description(f"Evaluating on test dataset ({corpus_bleu(translations, [targets]).score})")

    print(corpus_bleu(translations, [targets]))

main()