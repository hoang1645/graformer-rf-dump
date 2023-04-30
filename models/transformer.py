import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union


class CustomGraformer(nn.Module):
    def __init__(self, masked_encoder:Union[PreTrainedModel, str], causal_decoder: Union[PreTrainedModel, str], 
                 d_model:int=512, n_heads=8, dff=2048, n_encoder_layers=6, n_decoder_layers=6,
                 layer_norm=1e-5, dropout=.1, activation=F.gelu, encoder_tokenizer=None, decoder_tokenizer=None, device='cuda',
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
        The two models should be of the same size        

        For the additional K-layer encoder-decoder stack:
        :param d_model: dimension of the model, an int
        :param n_heads: number of attention heads
        :param dff: dimension of the feed-forward network
        :param n_encoder_layers: number of encoder layers
        :param n_decoder_layers: number of decoder layers

        d_model, n_heads and dff has to match the masked encoder and causal decoder.

        General parameters for the additional stacks:
        :param layer_norm: Layer normalization hyperparameter
        :param dropout: Rate of dropout
        :param activation: Activation function used, default to gelu

        General parameters for the entire model:
        :param encoder_tokenizer: Tokenizer used for the encoder. If `None`, `masked_encoder` must be `str`.
        :param decoder_tokenizer: Tokenizer used for the decoder. If `None`, `causal_decoder` must be `str`.
        """

        super().__init__(*args, **kwargs)

        if isinstance(masked_encoder, str):
            self.masked_encoder = AutoModel.from_pretrained(masked_encoder)
        else:
            self.masked_encoder = masked_encoder

        if isinstance(causal_decoder, str):
            self.causal_decoder = AutoModel.from_pretrained(causal_decoder)
        else:
            self.causal_decoder = causal_decoder

        # Freeze the causal decoder
        if isinstance(causal_decoder, str):
            self.causal_decoder.requires_grad_(False)
            self.masked_encoder.requires_grad_(False)
        else:
            for child in list(self.causal_decoder.children())[1:]:
                child.requires_grad_(False)

        self.k_layer_encoder_stack = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, dff, dropout, activation, layer_norm_eps=layer_norm, batch_first=True),
            # norm = nn.LayerNorm(self.masked_encoder.config.vocab_size, layer_norm),
            num_layers=n_encoder_layers
        )

        self.k_layer_decoder_stack = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, n_heads, dff, dropout, activation, layer_norm_eps=layer_norm, batch_first=True),
            # norm=nn.LayerNorm(self.causal_decoder.config.vocab_size, layer_norm),
            num_layers=n_decoder_layers
        )

        self.device = device

        # HuggingFace will raise errors itself if both tokenizer and model are None.
        self.encoder_tokenizer = AutoTokenizer.from_pretrained(masked_encoder) \
            if encoder_tokenizer is None else encoder_tokenizer
        

        # Defining special tokens if missing
        if self.encoder_tokenizer.bos_token_id is None:
            self.encoder_tokenizer.bos_token_id = self.encoder_tokenizer.unk_token_id
            self.encoder_tokenizer.bos_token = self.encoder_tokenizer.unk_token

        if self.encoder_tokenizer.eos_token_id is None:
            self.encoder_tokenizer.eos_token_id = self.encoder_tokenizer.unk_token_id
            self.encoder_tokenizer.eos_token = self.encoder_tokenizer.unk_token

        if self.encoder_tokenizer.pad_token_id is None:
            self.encoder_tokenizer.pad_token_id = self.encoder_tokenizer.eos_token_id
            self.encoder_tokenizer.pad_token = self.encoder_tokenizer.eos_token

        self.decoder_tokenizer = AutoTokenizer.from_pretrained(causal_decoder) \
            if decoder_tokenizer is None else decoder_tokenizer

        if self.decoder_tokenizer.bos_token_id is None:
            self.decoder_tokenizer.bos_token_id = self.decoder_tokenizer.unk_token_id
            self.decoder_tokenizer.bos_token = self.decoder_tokenizer.unk_token

        if self.decoder_tokenizer.eos_token_id is None:
            self.decoder_tokenizer.eos_token_id = self.decoder_tokenizer.unk_token_id
            self.decoder_tokenizer.eos_token = self.decoder_tokenizer.unk_token

        if self.decoder_tokenizer.pad_token_id is None:
            self.decoder_tokenizer.pad_token_id = self.decoder_tokenizer.eos_token_id
            self.decoder_tokenizer.pad_token = self.decoder_tokenizer.eos_token
            
        self.lmhead = nn.Linear(d_model, self.decoder_tokenizer.vocab_size)
            
    def forward(self, source, src_mask, target, tgt_mask) -> torch.Tensor:
        masked_encoder_output = self.masked_encoder(source, src_mask).last_hidden_state
        
        # decoder_tokens = self.decoder_tokenizer(target, return_tensors='pt', padding='max_length', max_length=128)
        # decoder_input_ids, decoder_attention_mask = decoder_tokens['input_ids'], decoder_tokens['attention_mask']
        causal_decoder_output = self.causal_decoder(target, attention_mask=tgt_mask).last_hidden_state

        memory = self.k_layer_encoder_stack.forward(masked_encoder_output, src_key_padding_mask=(1-src_mask).type(torch.bool),
                                                    mask=torch.zeros((source.shape[-1], source.shape[-1])).type(torch.bool).to('cuda'))

        output = self.k_layer_decoder_stack.forward(causal_decoder_output, memory, tgt_key_padding_mask=(1-tgt_mask).type(torch.bool),
                                                    tgt_mask=__class__.generate_square_subsequent_mask(target.shape[-1]).to(dtype=causal_decoder_output.dtype))


        return self.lmhead(output + causal_decoder_output)
    
    def encode(self, source, src_mask):
        masked_encoder_output = self.masked_encoder(source, src_mask).last_hidden_state
        return self.k_layer_encoder_stack.forward(masked_encoder_output, src_key_padding_mask=(1-src_mask).type(torch.bool),
                                                  mask=torch.zeros((source.shape[-1], source.shape[-1])).type(torch.bool).to('cuda'))

    def decode(self, tgt, memory, tgt_mask):
        causal_decoder_output = self.causal_decoder(tgt, attention_mask=tgt_mask).last_hidden_state
        return causal_decoder_output, self.k_layer_decoder_stack.forward(causal_decoder_output, memory,
                                                                         tgt_key_padding_mask=(1-tgt_mask).type(torch.bool),
                                                                         tgt_mask=__class__.generate_square_subsequent_mask(tgt.shape[-1]).to(dtype=causal_decoder_output.dtype))
    
    @staticmethod
    def generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones((sz, sz), device='cuda')) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    

    # @staticmethod
    # def create_mask(src, tgt, device='cpu'):
    #     src_seq_len = src.shape[0]
    #     tgt_seq_len = tgt.shape[0]
    #     DEVICE = device
    #     tgt_mask = __class__.generate_square_subsequent_mask(tgt_seq_len)
    #     src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    #     src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    #     tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    #     return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def greedy_decode(self, src, src_mask, max_len, start_symbol):
        DEVICE = self.device
        src = src.to(DEVICE)
        src_mask = src_mask.to(DEVICE)

        self.eval()

        memory = self.encode(src, src_mask)

        num_of_sentences = src.shape[0]
        
        last_word = torch.zeros((num_of_sentences, 1)).long()

        ys = torch.ones(num_of_sentences, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
        for _ in range(max_len-1):
            
            memory = memory.to(DEVICE)
            tgt_mask = torch.ones_like(ys).to(DEVICE)
            causal_out, out = self.decode(ys, memory, tgt_mask)
            # print(causal_out.shape, out.shape)
            # out = out.transpose(0, 1)
            prob = self.lmhead(causal_out[:,-1] + out[:, -1])
            
            _, next_word = torch.max(prob, dim=-1)
            next_word = next_word.reshape((-1,1))
            last_word = next_word.long().reshape((-1,1))
            ys = torch.cat([ys, \
                            last_word], dim=1)
            if all(next_word==self.decoder_tokenizer.pad_token_id): break
        # print(ys)
        return ys
    
    def translate(self, sentences:str):
        encoder_tokens = self.encoder_tokenizer(sentences, return_tensors='pt', padding=True)
        
        encoder_input_ids, encoder_attention_mask = encoder_tokens.input_ids, encoder_tokens.attention_mask
        target_tokens = self.greedy_decode(encoder_input_ids, encoder_attention_mask, max_len=128, start_symbol=self.decoder_tokenizer.bos_token_id)
        return self.decoder_tokenizer.batch_decode(target_tokens, skip_special_tokens=True)

# tokenizer = SentencePieceTokenizer('sentencepiece.model')

# botched_bert, botched_gpt = get_model_with_different_embedding_layer('bert-base-uncased', 'openai-gpt', 
#                                                                         tokenizer.vocab_size, tokenizer.pad_token_id)


# model = CustomGraformer(
#         botched_bert, botched_gpt, 768,12,3072, encoder_tokenizer=tokenizer,
#         decoder_tokenizer=tokenizer
#     ).to('cuda')

# print(model.translate(['hello there']))
# x = torch.randint(1, 10000, size=(2, 30))
# x_mask = torch.ones(x.shape)

# y = torch.randint(1, 10000, size=(2, 35))
# y_mask = torch.ones(y.shape)

# print(model(x, x_mask, y, y_mask).shape)
