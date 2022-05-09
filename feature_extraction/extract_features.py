from wav2vec import Wav2Vec
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import glob
import traceback

def create_model(model_checkpoint):
    print("Creating model")
    wav_2_vec_model = Wav2Vec(model_checkpoint)
    wav_2_vec_model.eval()
    return wav_2_vec_model

def extract_wav2vec_features(model, langs):
    for lang in langs: # Process each language
        feats_dest_path = dest_path+lang
        os.makedirs(feats_dest_path, exist_ok=True)
        # Extract features for test files
        csv_path = csv_base_path + f"{lang}_test.csv"
        all_lines = pd.read_csv(csv_path)
        for idx in tqdm(range(len(all_lines))):
            filename = all_lines["filename"].iloc[idx]
            wav_path = wav_base_path + lang + "/" + filename
            features_path = feats_dest_path + "/" + wav_path.split("/")[-1].split(".")[0] + ".feats"
            try:
                wav_feats = model.extract_features(wav_path)
                # Move features to CPU and save as numpy
                np.save(features_path, wav_feats.cpu().detach().numpy())
            except Exception as err:
                print(f"Error while processing Wav file: {wav_path}")
                print(str(err))
                print(traceback.format_exc())

        # Extract features for train files
        csv_path = csv_base_path + f"{lang}_train.csv"
        all_lines = pd.read_csv(csv_path)
        for idx in tqdm(range(len(all_lines))):
            filename = all_lines["filename"].iloc[idx]
            wav_path = wav_base_path + lang + "/" + filename
            features_path = feats_dest_path + "/" + wav_path.split("/")[-1].split(".")[0] + ".feats"
            try:
                wav_feats = model.extract_features(wav_path)
                # Move features to CPU and save as numpy
                np.save(features_path, wav_feats.cpu().detach().numpy())
            except Exception as err:
                print(f"Error while processing Wav file: {wav_path}")
                print(str(err))
                print(traceback.format_exc())

def extract_features(model_checkpoint):
    wav_2_vec_model = create_model(model_checkpoint)
    extract_wav2vec_features(wav_2_vec_model, langs)


wav_base_path = "./SC_audio_" # Path to the WAV files
csv_base_path = "./annotations/" # Path to the CSV files
langs = ["Hindi"] # List of languages that need to be processed
dest_path = "./features_output/" # Path to the Extracted Features
# XLSR checkpoint - https://github.com/pytorch/fairseq/tree/main/examples/wav2vec#wav2vec-20
#model_checkpoint = "./xlsr_53_56k.pt" 
# CLSRIL checkpoint - https://github.com/Open-Speech-EkStep/vakyansh-models
model_checkpoint = "./CLSRIL-23.pt" 
extract_features(model_checkpoint)
