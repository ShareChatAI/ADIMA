import torch.nn as nn
import pdb
import torch 
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from transformers import AutoTokenizer, AutoModelForPreTraining
import soundfile as sf
import numpy as np
import fairseq

class Wav2Vec(nn.Module):
    
    def __init__(self, cls_type, load_model=False, dropout=0.1):
        super(Wav2Vec, self).__init__()
        self.pretrained_model = True
        self.cls_type = cls_type
        if cls_type == "gru":
            self.rnn = nn.GRU(512, 512, 1, batch_first=True, bidirectional=True)
        elif cls_type == "lstm":
            self.rnn = nn.LSTM(512, 512, 1, batch_first=True, bidirectional=True)
        self.dropout = dropout
        self.init_classifier()
    
    def init_classifier(self):
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(128, 2),
            )
        
    def forward(self, x):
        if self.cls_type in ["fc_avg", "fc_max"]:
            x = self.classifier(x)    
        elif self.cls_type == "lstm":
            lstm_out, (ht, ct) = self.rnn(x)
            x = self.classifier(ht[-1])
        elif self.cls_type == "gru":
            gru_out, ht = self.rnn(x)
            x = self.classifier(ht[-1])
        return x
    

