#!/bin/zsh

CUDA_VISIBLE_DEVICES=0 python lightning_main.py \
--train_path_src parallel_datasets/en-vi/train.tags.en-vi.clean.en --train_path_tgt parallel_datasets/en-vi/train.tags.en-vi.clean.vi \
--valid_path_src parallel_datasets/en-vi/valid.en --valid_path_tgt parallel_datasets/en-vi/valid.vi \
--test_path_src parallel_datasets/en-vi/test.en --test_path_tgt parallel_datasets/en-vi/test.vi \
--batch_size 4 --epoch 12 --masked_encoder bert-base-uncased --causal_decoder gpt2 --tensor_core_precision medium