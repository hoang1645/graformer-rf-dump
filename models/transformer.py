import torch
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CustomTransformer(nn.Module):
    def __init__(self, encoder:nn.Module|str=None, decoder: nn.Module|str=None, encoder_embedding_size:int=None,
                 encoder_embedding_layer:nn.Module=None, encoder_d_model:int=512, encoder_n_heads=8, encoder_dff=2048, 
                 encoder_n_layers=6, decoder_embedding_size=None, decoder_embedding_layer:nn.Module=None,
                 decoder_d_model=512, decoder_n_heads=8, decoder_dff=2048, decoder_n_layers=6, 
                 layer_norm = 1e-5, dropout=.1, activation=F.gelu, encoder_tokenizer=None, decoder_tokenizer=None,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if isinstance(encoder, str):
            self.encoder = AutoModel.from_pretrained(encoder)
        elif isinstance(encoder, nn.Module):
            self.encoder = encoder
        else:
            self.encoder = nn.Sequential(
                # Embedding layer
                encoder_embedding_layer if encoder_embedding_layer is not None else nn.Embedding(encoder_embedding_size, encoder_d_model),
                # Encoder layer
                nn.TransformerEncoder(num_layers=encoder_n_layers, norm=nn.LayerNorm(encoder_d_model, layer_norm),
                    encoder_layer=nn.TransformerEncoderLayer(encoder_d_model, encoder_n_heads, encoder_dff, dropout, activation, layer_norm)
                )
            )

        if isinstance(decoder, str):
            self.decoder = AutoModelForCausalLM.from_pretrained(decoder)  # Model head included.
        elif isinstance(decoder, nn.Module):
            self.decoder = decoder  # Should include model head
        else:
            self.decoder = nn.Sequential(
                # Embedding layer
                decoder_embedding_layer if decoder_embedding_layer is not None else nn.Embedding(decoder_embedding_size, encoder_d_model),
                # Decoder layer
                nn.TransformerDecoder(num_layers=decoder_n_layers, norm=nn.LayerNorm(encoder_d_model, layer_norm),
                    encoder_layer=nn.TransformerDecoderLayer(decoder_d_model, decoder_n_heads, decoder_dff, dropout, activation, layer_norm)
                ),
                # Model head
                nn.LazyLinear(decoder_embedding_size)
            )

        self.transformer = nn.Transformer(
            custom_encoder=self.encoder, custom_decoder=self.decoder, layer_norm_eps=layer_norm
        )

        # HuggingFace will raise errors itself if both tokenizer and model are None.
        self.encoder_tokenizer = AutoTokenizer.from_pretrained(encoder) \
            if encoder_tokenizer is None else encoder_tokenizer
        self.decoder_tokenizer = AutoTokenizer.from_pretrained(decoder) \
            if decoder_tokenizer is None else decoder_tokenizer

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: Optional[torch.Tensor] = None, tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None, memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.transformer.forward(src, tgt, src_mask, tgt_mask, memory_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask)
        return F.softmax(out + tgt)
    
    def freeze_decoder(self):
        for param in self.decoder.parameters(): param.requires_grad=False