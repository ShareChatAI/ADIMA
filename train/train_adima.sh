#!/bin/bash

lang="Hindi"
src_lang=$lang
tgt_lang=$lang
workspace="./logs"
feats_base_path="./CLSRIL_features/"
cls_type="fc_avg"
mode="train"
csv_path="./annotations/"

CUDA_VISIBLE_DEVICES=0 python main.py \
--feats_base_path=$feats_base_path \
--workspace=$workspace \
--cls_type=$cls_type \
--src_lang=$src_lang \
--tgt_lang=$tgt_lang \
--mode=$mode \
--csv_path=$csv_path \
--max_epochs=50
