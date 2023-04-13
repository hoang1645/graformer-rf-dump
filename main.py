from models.transformer import CustomGraformer
from tqdm import tqdm
from datasets.dataloader import get_dataloader
import torch
import torch.nn.functional as F
from sacrebleu import corpus_bleu
from torchinfo import summary
from args.parser import GraformerArgumentParser
import os

def train(model:torch.nn.Module, train_dataloader:torch.utils.data.DataLoader, 
          val_dataloader:torch.utils.data.DataLoader, optim:torch.optim.Optimizer, epoch:int):
    min_val_loss = 999
    for e in range(epoch):
        # training
        model.train()
        losses = (0, 0)
        for train_batch in (bar:=tqdm(train_dataloader, desc=f"Epoch {e}")):
            x, y = train_batch
            x_input, x_mask = x.input_ids.to('cuda'), x.attention_mask.to('cuda')
            y_input, y_mask = y.input_ids[:, :-1].to('cuda'), y.attention_mask[:, :-1].to('cuda')

            out = model(x_input, x_mask, y_input, y_mask)
            y_out = y.input_ids[:, 1:].to('cuda')

            optim.zero_grad()

            loss = F.cross_entropy(out.reshape(-1, out.shape[-1]), y_out.reshape(-1))
            bar.set_description(f"Epoch {e}, loss = {loss}")
            losses = (losses[0] + loss.item(), losses[1] + 1)

            loss.backward()
            optim.step()

        print(f"Training loss: {losses[0]/losses[1]}")

        # validation
        losses = (0, 0)
        model.eval()
        for val_batch in val_dataloader(bar:=tqdm(train_dataloader, desc=f"Evaluating")):
            x, y = val_batch
            x_input, x_mask = x.input_ids.to('cuda'), x.attention_mask.to('cuda')
            y_input, y_mask = y.input_ids[:, :-1].to('cuda'), y.attention_mask[:, :-1].to('cuda')

            out = model(x_input, x_mask, y_input, y_mask)
            y_out = y.input_ids[:, 1:].to('cuda')

            loss = F.cross_entropy(out.reshape(-1, out.shape[-1]), y_out.reshape(-1))
            losses = (losses[0] + loss.item(), losses[1] + 1)
        
        print(f"Validation loss: {losses[0]/losses[1]}")
        if losses[0]/losses[1] < min_val_loss:
            min_val_loss = losses[0]/losses[1]
            torch.save(model, "outputs/best.pt")
        torch.save(model, f"checkpoint_{e}.pt")

def main():
    # torch._dynamo.config.suppress_errors = True
    parser = GraformerArgumentParser()
    args = parser.get_args()

    model = CustomGraformer(
            args.masked_encoder, args.causal_decoder, args.d_model, args.n_heads, args.dff,
        ).to('cuda')
    # if os.name != 'nt': model = torch.compile(model, backend='inductor', mode='reduce-overhead')
    summary(model)
    
    if args.from_checkpoint:
        model.load_state_dict(torch.load(args.from_checkpoint))

    if not args.test_only:
        optim = torch.optim.Adagrad(model.parameters(), args.lr, 1e-5, 1e-5)
        # dataloader goes here
        train_dataloader = get_dataloader(args.train_path_src, args.train_path_tgt, 
                                          model.encoder_tokenizer, 
                                          model.decoder_tokenizer,
                                          batch_size=args.batch_size, num_workers=args.num_workers)
        val_dataloader = get_dataloader(args.valid_path_src, args.valid_path_tgt, 
                                          model.encoder_tokenizer, 
                                          model.decoder_tokenizer,
                                          batch_size=args.batch_size, num_workers=args.num_workers)
        
        if not os.path.isdir('outputs'): os.mkdir('outputs')
        train(model, train_dataloader, val_dataloader, optim, args.epoch)
    
    test_dataloader  = get_dataloader(args.test_path_src, args.test_path_tgt, 
                                          model.encoder_tokenizer, 
                                          model.decoder_tokenizer,
                                          batch_size=args.batch_size, num_workers=args.num_workers, test=True) # don't tokenize in test dataloader    
    translations = []
    targets = []

    for src, tgt in tqdm(test_dataloader):
        trl = model.translate(src)
        translations.extend(trl)
        targets.extend(tgt)
    
    print(corpus_bleu(translations, [targets]))

main()