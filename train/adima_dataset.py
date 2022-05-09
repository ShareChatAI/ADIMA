from torch.utils.data import DataLoader, Dataset, random_split
import torchaudio
from audio_util import AudioUtil
import os
import pandas as pd
import numpy as np
import torch 
import pdb

class Adima(Dataset):
    
    def __init__(self, args, filepath, feats_wav_path, split):
        self.csv_data = pd.read_csv(filepath)
        self.feats_wav_path = feats_wav_path
        self.split = split
        self.args = args
        self.cls_type = args.cls_type
        self.duration = args.duration 
        self.sample_rate = args.sample_rate
        self.shift_pct = args.shift_pct
        self.class_map = {"Yes":1, "No":0}

    def __len__(self):
        return len(self.csv_data)    
    
    def __getitem__(self, idx):
        
        audio_file_name = self.csv_data["filename"].iloc[idx]
        class_name = self.csv_data["label"].iloc[idx]
        class_id = self.class_map[class_name]
        # pdb.set_trace()
        # Load features  
        audio_file_name = audio_file_name.split("/")[-1].split(".")[0]
        feats_path = f"{self.feats_wav_path}/{audio_file_name}.feats.npy"

        if not os.path.exists(feats_path):
            return None

        feats = np.load(feats_path) 
        feats = feats.squeeze()
        feats = torch.tensor(feats)

        if self.split == "train": # Augmentation    
            feats, sr = AudioUtil.time_shift((feats, self.args.sample_rate), self.shift_pct)
            feats = feats.unsqueeze(0)
            feats = AudioUtil.spectro_augment(feats, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
            feats = feats.squeeze()
        
        if self.cls_type == "fc_avg":
            feats = feats.mean(axis=1) # Squeeze temporal dimension
        elif self.cls_type == "fc_max":
            feats = torch.max(feats, axis=1)[0] 
        else:
            # RNNs
            pass
        
        return feats, class_id
        