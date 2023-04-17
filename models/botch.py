from transformers import BertModel, OpenAIGPTModel, BertConfig, OpenAIGPTConfig
from typing import OrderedDict

def get_model_with_different_embedding_layer(bert_name, gpt_name='openai-gpt', vocab_size=64003, pad_token_id=2):
    original_bert = BertModel.from_pretrained(bert_name).to('cpu')

    botched_bert_config = BertConfig.from_pretrained(bert_name)
    botched_bert_config.vocab_size = vocab_size
    botched_bert_config.pad_token_id = pad_token_id
    print(botched_bert_config)

    botched_bert = BertModel(botched_bert_config).to('cpu')
    bert_keepers = OrderedDict()
    bert_keepers_keys = list(original_bert.state_dict().keys())
    bert_keepers_keys.remove('embeddings.word_embeddings.weight')

    for key in bert_keepers_keys:
        bert_keepers[key] = original_bert.state_dict()[key]
        
    bert_keepers['embeddings.word_embeddings.weight'] = botched_bert.state_dict()['embeddings.word_embeddings.weight']
    botched_bert.load_state_dict(bert_keepers)

    original_gpt = OpenAIGPTModel.from_pretrained(gpt_name).to('cpu')
    botched_gpt_config = OpenAIGPTConfig.from_pretrained(gpt_name)
    botched_gpt_config.vocab_size = vocab_size
    botched_gpt_config.pad_token_id = pad_token_id
    print(botched_gpt_config)
    botched_gpt = OpenAIGPTModel(botched_gpt_config).to('cpu')

    gpt_keepers = OrderedDict()
    gpt_keepers_keys = list(original_gpt.state_dict().keys())
    gpt_keepers_keys.remove('tokens_embed.weight')

    for key in gpt_keepers_keys:
        gpt_keepers[key] = original_gpt.state_dict()[key]

    gpt_keepers['tokens_embed.weight'] = botched_gpt.state_dict()['tokens_embed.weight']
    botched_gpt.load_state_dict(gpt_keepers)
    return botched_bert, botched_gpt
