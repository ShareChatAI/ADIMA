import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))
import numpy as np
import h5py
import argparse
import time
from sklearn import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve
from adima_dataset import Adima
from sklearn.metrics import balanced_accuracy_score
import pdb
from tqdm import tqdm


def pad_subsample(sample, pad_to=20):
    ''' pad to certain length (len<pad_to) or 
    subsample at equal intervals (len>pad_to) for temporal models'''
    s_=sample.shape
    if s_[1]<pad_to:
        a=np.zeros((s_[0],pad_to))
        a[:, :s_[1]]=sample
    elif s_[1]>pad_to:
        dist=s_[1]//pad_to
        a=sample[:, :pad_to*dist:dist]
    else:
        return sample.numpy()
    return a

def collate_temporal(batch):
    batch = sorted(batch, key=lambda x: x[0].shape[1], reverse=True)
    labels = []
    for sample in batch:
        labels.append(sample[1])
    max_duration = batch[0][0].shape[1]
    padded_samples = [pad_subsample(sample[0], max_duration) for sample in batch]
    
    acoustic = torch.FloatTensor(padded_samples).transpose(1,2)
    return acoustic, torch.tensor(labels)

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def train_epoch(args, epoch, model, train_dl, criterion, optimizer):
    
    model.train()
    
    running_loss = 0.0
    correct_prediction = 0
    total_prediction = 0

    for (batch_x, batch_y) in tqdm(train_dl):
        
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        
        optimizer.zero_grad()
        outputs = model(batch_x)
    
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, prediction = torch.max(outputs, 1)
        # Count of predictions that matched the target label
        correct_prediction += (prediction == batch_y).sum().item()
        total_prediction += prediction.shape[0]
    
    num_batches = len(train_dl)
    avg_loss = running_loss / num_batches
    acc = correct_prediction/total_prediction
    print(f'Training: Epoch: {epoch}, Loss: {avg_loss:.2f}, Train Accuracy: {acc:.2f}, Num Samples: {len(train_dl)}')

def inference(args, model, test_dl):

    model.eval()
    correct_prediction = 0
    total_prediction = 0
    pred = []
    true_lab = []
    print("Inference ==>")
    
    with torch.no_grad():    
        
        for data in tqdm(test_dl):
           
            inputs, labels = data[0].cuda(), data[1].cuda()    
            outputs = model(inputs)
            _, prediction = torch.max(outputs, 1)

            for j in range(prediction.shape[0]):
                pred.append(prediction[j].item())
                true_lab.append(labels[j].item())
            
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]
        
        acc = correct_prediction/total_prediction
        
        f1_macro = f1_score(true_lab, pred, average='macro')
        f1_micro = f1_score(true_lab, pred, average='micro')
            
        print(f'Test: Accuracy: {acc}, F1_Macro: {f1_macro}, F1_Micro: {f1_micro} Total samples: {total_prediction}')
        print(50*"-")
        return acc, f1_macro, f1_micro

def train(args, model):
    
    model.cuda()
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-07)
    
    os.makedirs("checkpoints", exist_ok=True)
    
    ds_train = Adima(args, args.train_csv_path, args.train_feat_wav_path, split="train")
    ds_test = Adima(args, args.test_csv_path, args.test_feat_wav_path, split="test")
    criterion = nn.CrossEntropyLoss()
    
    coll_fn = collate_fn
    if args.cls_type in ["gru", "lstm"]:
        coll_fn = collate_temporal
        
    train_dl = torch.utils.data.DataLoader(ds_train, batch_size = args.batch_size, shuffle = True, collate_fn = coll_fn)
    test_dl = torch.utils.data.DataLoader(ds_test, batch_size = args.batch_size, shuffle = False, collate_fn = coll_fn)

    model.train()
    best_test_acc = 0
    best_epoch = 0
    best_test_f1 = 0
    
    for epoch in range(args.max_epochs):
        
        print(f"Epoch ==> {epoch}")
        
        train_epoch(args, epoch, model, train_dl, criterion, optimizer)
        
        test_acc_epoch, test_f1_macro, _ = inference(args, model, test_dl)
        
        if test_acc_epoch > best_test_acc:
            best_test_acc = test_acc_epoch
            best_test_f1 = test_f1_macro
            best_epoch = epoch
            model_path = f"checkpoints/{args.src_lang}_epoch_{best_epoch}_clstype_{args.cls_type}.pt"
            print(f"Saving model ==> {model_path}")
            torch.save(model, model_path)
    
    print("Finishing up ==> ")
    rfh = open(f"checkpoints/{args.src_lang}_{args.tgt_lang}_epoch_{best_epoch}_clstype_{args.cls_type}.out", "w")
    rfh.write(str(best_epoch)+"_"+str(best_test_acc)+"_"+str(best_test_f1))
    print(f"Best Epoch {best_epoch}")
    print(f"Best Acc {best_test_acc}")
    print(f"Best F1 Macro {best_test_f1}")