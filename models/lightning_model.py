import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from models.transformer import CustomGraformer

class LightningGraformer(pl.LightningModule):
    def __init__(self, masked_encoder:nn.Module|str, causal_decoder: nn.Module|str, 
                 d_model:int=512, n_heads=8, dff=2048, n_encoder_layers=6, n_decoder_layers=6,
                 layer_norm=1e-5, dropout=.1, activation=F.gelu, encoder_tokenizer=None, decoder_tokenizer=None,
                 lr=1e-5,
                 *args, **kwargs) -> None:
        """
        Implementation of Graformer, as per the article
        "Multilingual Translation via Grafting Pre-trained Language Models", Zewei et al.,
        Findings of the Association for Computational Linguistics: EMNLP 2021, pages 2735-2747, November 7-11, 2021

        Parameters:
        :param masked_encoder: the masked LM encoder, mBERT in the original paper. A string for the name of the model
            as per Hugging Face's naming rules, or the model inherited by the Hugging Face's PreTrainedModel.
        :param causal_decoder: the causal decoder, mGPT in the original paper. A string for the name of the model
            as per Hugging Face's naming rules, or the model inherited by the Hugging Face's PreTrainedModel.
        
        The two models should be of the same size.        

        For the additional K-layer encoder-decoder stack:
        :param d_model: dimension of the model, an int
        :param n_heads: number of attention heads
        :param dff: dimension of the feed-forward network
        :param n_encoder_layers: number of encoder layers
        :param n_decoder_layers: number of decoder layers

        `d_model`, `n_heads` and `dff` have to match the masked encoder and causal decoder.

        General parameters for the additional stacks:
        :param layer_norm: Layer normalization hyperparameter
        :param dropout: Rate of dropout
        :param activation: Activation function used, default to gelu

        General parameters for the entire model:
        :param encoder_tokenizer: Tokenizer used for the encoder. If `None`, `masked_encoder` must be `str`.
        :param decoder_tokenizer: Tokenizer used for the decoder. If `None`, `causal_decoder` must be `str`.
        """
        super().__init__()
        self.graformer = CustomGraformer(masked_encoder, causal_decoder, 
                 d_model, n_heads, dff, n_encoder_layers, n_decoder_layers,
                 layer_norm, dropout, activation, encoder_tokenizer, decoder_tokenizer,
                 *args, **kwargs)
        # The power of torch 2.0
        #  self.graformer = torch.compile(self.graformer, backend='inductor')
        self.lr = lr

        self.last_val_loss = 1000
        self.curr_val_loss = 0
        
        
    def forward(self, source, src_mask, target, tgt_mask):
        self.graformer.forward(source, src_mask, target, tgt_mask)

    def configure_optimizers(self):
        return torch.optim.Adagrad(self.graformer.parameters(), self.lr)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x_input, x_mask = x.input_ids, x.attention_mask
        y_input, y_mask = y.input_ids[:, :-1], y.attention_mask[:, :-1]

        out = self.graformer(x_input, x_mask, y_input, y_mask)
        y_out = y.input_ids[:, 1:]
        loss = F.cross_entropy(out.reshape(-1, out.shape[-1]), y_out.reshape(-1))

        self.log("loss", loss)
        return loss
    
    def validation_step(self, valid_batch, valid_idx):
        x, y = valid_batch
        x_input, x_mask = x.input_ids, x.attention_mask
        y_input, y_mask = y.input_ids[:, :-1], y.attention_mask[:, :-1]

        out = self.graformer(x_input, x_mask, y_input, y_mask)
        y_out = y.input_ids[:, 1:]
        val_loss = F.cross_entropy(out.reshape(-1, out.shape[-1]), y_out.reshape(-1))

        self.log("val_loss", val_loss)
        self.curr_val_loss = val_loss
        return val_loss
    
    
    def on_validation_epoch_end(self) -> None:
        if self.curr_val_loss < self.last_val_loss:
            torch.save(self.graformer.state_dict(), 'outputs/checkpoint_best.pt')
        self.last_val_loss = self.curr_val_loss