import torch.nn as nn
import pdb
import torch 
import soundfile as sf
import numpy as np
import fairseq
import librosa

class Wav2Vec(nn.Module):
    
    def __init__(self, checkpoint):
        super(Wav2Vec, self).__init__()
        self.__load_model__(checkpoint)

    def __load_model__(self, checkpoint):
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint])
        self.model = model[0]
        self.model.eval()

    def extract_features(self, wav_file_path):
        audio_input, sample_rate = librosa.load(wav_file_path, sr=16000)  # sf.read(wav_file_path)
        audio_input = torch.tensor(audio_input).unsqueeze(0).float()
        return self.model.feature_extractor(audio_input)