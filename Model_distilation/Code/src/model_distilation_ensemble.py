import sys
import time
import h5py

import numpy as np
import re
from math import ceil
from sklearn.metrics import average_precision_score
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import pickle
#import pickle5 as pickle
from tqdm import tqdm
import torch.nn.functional as F

from sklearn.model_selection import train_test_split

from scipy.sparse import load_npz
from glob import glob

from transformers import get_constant_schedule_with_warmup
from sklearn.metrics import precision_score,recall_score,accuracy_score
import copy

from train import trainModel
from dataloader import getData,spliceDataset,h5pyDataset,getDataPointList,getDataPointListFull,DataPointFull
from weight_init import keras_init
from losses_2 import categorical_crossentropy_2d, SpliceDistillationLoss
from model import SpliceFormer, SpliceFormerLite
from evaluation_metrics import print_topl_statistics,cross_entropy_2d
from gpu_metrics import run_bootstrap

from collections import OrderedDict

#----------
print("Starting here- ENSEMBLE")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Selected device: {device}")

rng = np.random.default_rng(23673)

data_dir = '../../Data'
setType = 'train'
annotation, transcriptToLabel, seqData = getData(data_dir, setType)

train_gene, validation_gene = train_test_split(annotation['gene'].drop_duplicates(),test_size=.1,random_state=435)
annotation_train = annotation[annotation['gene'].isin(train_gene)]
annotation_validation = annotation[annotation['gene'].isin(validation_gene)]

#print(annotation_validation)

print("Defining hyperparameters")

L = 32
N_GPUS = 1
k = 2
NUM_ACCUMULATION_STEPS=1

# Hyper-parameters:
# L: Number of convolution kernels
# W: Convolution window size in each residual unit
# AR: Atrous rate in each residual unit

W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                21, 21, 21, 21, 41, 41, 41, 41])

AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                 10, 10, 10, 10, 25, 25, 25, 25])

Wlite = np.asarray([11, 11, 11, 11, 11, 11, 11, 11])
ARlite = np.asarray([1, 1, 1, 1, 4, 4, 4, 4])

BATCH_SIZE = 16*k*N_GPUS

k = NUM_ACCUMULATION_STEPS*k

CL_teacher = 2 * np.sum(AR*(W-1))
CL_student = 2 * np.sum(ARlite*(Wlite-1))

SL=5000
T_CL_MAX=40000
S_CL_MAX = 20000
T = 5.0    #Temperature
ALPHA = 0.9

print("Processing data")

train_dataset = spliceDataset(getDataPointListFull(annotation_train,transcriptToLabel,SL,S_CL_MAX,shift=SL))
val_dataset = spliceDataset(getDataPointListFull(annotation_validation,transcriptToLabel,SL,S_CL_MAX,shift=SL))
train_dataset.seqData = seqData
val_dataset.seqData = seqData

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=10, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=10)
print(val_loader)

epochs = 10 #30 IS IMPOSSIBLE
hs = []
learning_rate= k*1e-3
gamma=0.5

print("professor parameters")

TEACHER_PATHS = [
    '../../Results/PyTorch_Models/transformer_encoder_45k_finetune_rnasplice-blood_all_050623_0',
    '../../Results/PyTorch_Models/transformer_encoder_45k_finetune_rnasplice-blood_all_050623_1',
    '../../Results/PyTorch_Models/transformer_encoder_45k_finetune_rnasplice-blood_all_050623_2',
    '../../Results/PyTorch_Models/transformer_encoder_45k_finetune_rnasplice-blood_all_050623_3',
    '../../Results/PyTorch_Models/transformer_encoder_45k_finetune_rnasplice-blood_all_050623_4',
    '../../Results/PyTorch_Models/transformer_encoder_45k_finetune_rnasplice-blood_all_050623_5',
    '../../Results/PyTorch_Models/transformer_encoder_45k_finetune_rnasplice-blood_all_050623_6',
    '../../Results/PyTorch_Models/transformer_encoder_45k_finetune_rnasplice-blood_all_050623_7',
    '../../Results/PyTorch_Models/transformer_encoder_45k_finetune_rnasplice-blood_all_050623_8',
    '../../Results/PyTorch_Models/transformer_encoder_45k_finetune_rnasplice-blood_all_050623_9',
]
T_PARAMS = {
    'CL_max': T_CL_MAX, 'n_channels': 32, 'depth': 4, 
    'heads': 4, 'n_transformer_blocks': 2, 'determenistic': True, 
    'crop': True, 'bn_momentum': 0.01 / NUM_ACCUMULATION_STEPS
}

teacher_models = []
print("Loading Ensemble of Professors...")

try:
   for i, path in enumerate(TEACHER_PATHS):
        teacher_model = SpliceFormer(**T_PARAMS)
        teacher_model.to(device)

        state_dict = torch.load(path, map_location=device)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] #remove `module.`
            new_state_dict[name] = v
            
        teacher_model.load_state_dict(new_state_dict)
        teacher_model.eval() 
        for param in teacher_model.parameters():
            param.requires_grad = False
        
        teacher_models.append(teacher_model)
        print(f"  Teacher {i+1} loaded from {path.split('/')[-1]}")

except Exception as e:
    print(f"ERROR Professor: {e}")
    sys.exit()

print("Student model initialization")

S_PARAMS = {
    'CL_max': S_CL_MAX, 'n_channels': 16, 'depth': 2, 
    'heads': 2, 'n_transformer_blocks': 1, 'determenistic': True, 
    'crop': True, 'bn_momentum': 0.01 / NUM_ACCUMULATION_STEPS
}

try:
    student_model = SpliceFormerLite(**S_PARAMS)
    student_model.to(device)
    
    print("Student (SpliceFormerLite) instanciated.")

except Exception as e:
    print(f"ERROR: {e}")
    sys.exit()

#--- NEW: Class Weighting ---
#Inverse Weights: W_Null, W_Acceptor, W_Donor.

#Null should have a low weight (1) and Acceptor/Donor a high weight (10000)
#The site density is very low.
CLASS_WEIGHTS = torch.tensor([1.0, 1000.0, 1000.0], device=device).float()

#SpliceDistillationLoss combine KL-Divergence (Soft) e Cross-Entropy (Hard)
distillation_loss_fn = SpliceDistillationLoss(T=T, alpha=ALPHA, class_weights=CLASS_WEIGHTS)

#Optimizer: ONly student should be trained.
optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate, weight_decay=1e-5)

#--- NEW: Learning and Warm-up Rate Scheduler ---
#Scheduler to reduce LR after 5 seasons 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)
# Warmup for the first 5 epochs
warmup = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=int(5*len(train_loader)/NUM_ACCUMULATION_STEPS))
#--- END NEW: Scheduler ---

print(f"Config: Distilation loss(T={T}, Alpha={ALPHA}).")


EPOCHS = 10 #30 IS IMPOSSIBLE

print(student_model.train()) 
print(teacher_model.eval())

history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}
best_val_loss = float('inf')
best_student_weights = copy.deepcopy(student_model.state_dict())
checkpoint_path='best_spliceformerlite_distillation_ENSEMBLE.pt'

print("\n--- CHECKING SHAPES (Epoch 1) ---")

for batch_idx, (input_data, hard_targets) in enumerate(train_loader):
    if batch_idx >=5:
        break

    input_data = input_data.to(device).float()
    hard_targets = hard_targets.to(device).float()

    optimizer.zero_grad()

    teacher_output = teacher_model(input_data, distillation_mode=True)
    teacher_logits = teacher_output[0]

    teacher_logits = teacher_logits.detach() 

    student_output = student_model(input_data)
    student_logits = student_output[0]

    L_FULL = teacher_logits.shape[2]
    L_TARGET = student_logits.shape[2]
    start = int((L_FULL - L_TARGET)/2)
    end = int(start + L_TARGET)

    teacher_logits_cropped = teacher_logits[:, :, start:end]
    hard_targets_cropped = hard_targets[:, :, start:end]
    
    print(f"\nBatch {batch_idx}")
    print(f"Shape Teacher logits {teacher_logits_cropped.shape}")
    print(f"Shape Student logits {student_logits.shape}")
    
    print(f"Shape hard tardes {hard_targets_cropped.shape}")

    try:
        loss = distillation_loss_fn(student_logits, teacher_logits_cropped, hard_targets_cropped)
        print(f"results: {loss.item():.4f} (ok)")

        loss.backward()
        optimizer.step()

    except RuntimeError as e:
        print(f"ERRO NO {batch_idx}")
        print(f"Detalhe do erro: {e}")

for epoch in range(1, EPOCHS + 1):
    
    #--- Train ---
    student_model.train()
    total_train_loss = 0.0
    start_time = time.time()
    
    train_loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} (Train)", dynamic_ncols=True)
    
    for i, (input_data, hard_targets) in enumerate(train_loop):
        input_data = input_data.to(device).float()
        hard_targets = hard_targets.to(device).float()

        optimizer.zero_grad()

        #Professor output(Soft Targets)
        #We use with torch.no_grad() for efficiency and to ensure the gradient graph...
        
        with torch.no_grad():
            teacher_logits_list = []
            for teacher in teacher_models:
                teacher_output = teacher(input_data, distillation_mode=True)
                teacher_logits_list.append(teacher_output[0])

            teacher_logits_full = torch.mean(torch.stack(teacher_logits_list), dim=0) # [B, C, L_FULL]
        #Student output
        student_output = student_model(input_data)
        student_logits = student_output[0] # [B, C, L_TARGET]

        #Alignment and Cutting of Professor Logits and Hard Targets
        L_TARGET = student_logits.shape[2] 
        L_FULL = teacher_logits_full.shape[2] 
        
        start = int((L_FULL - L_TARGET) / 2)
        end = int(start + L_TARGET)
        
        teacher_logits_cropped = teacher_logits_full[:, :, start:end].detach() 
        hard_targets_cropped = hard_targets[:, :, start:end]

        #Calculation of Distillation Loss
        loss = distillation_loss_fn(student_logits, 
                                    teacher_logits_cropped, 
                                    hard_targets_cropped)
        
        #Backpropagation and Optimization
        loss.backward()
        optimizer.step()

        if epoch <= 5: 
            warmup.step()

        total_train_loss += loss.item() * input_data.size(0) 
        train_loop.set_postfix(loss=loss.item())

    avg_train_loss = total_train_loss / len(train_loader.dataset)
    history['train_loss'].append(avg_train_loss)
    
    if epoch > 5:
        scheduler.step()
  
    #validation
    student_model.eval()
    total_val_loss = 0.0
    val_logits_list = []
    val_targets_list = []

    with torch.no_grad():
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} (Val)", dynamic_ncols=True)
        for input_data, hard_targets in val_loop:
            input_data = input_data.to(device).float()
            hard_targets = hard_targets.to(device).float()

            #Student output
            student_output = student_model(input_data)
            student_logits = student_output[0]
            
            #For cutting off hard targets (using Teacher's CL_max for offset calculation)
            L_TARGET = student_logits.shape[2] 
            L_input = input_data.shape[2]
            
            #The size of the Teacher's output would be L_input - T_CL_MAX.
            L_FULL_FAKE = L_input - T_CL_MAX
            
            start = int((L_FULL_FAKE - L_TARGET) / 2)
            end = int(start + L_TARGET)
            
            #CUtting dos Hard Targets
            hard_targets_cropped = hard_targets[:, :, start:end]

            #Validation Loss: Hard Loss (Cross-Entropy) Only
            #Using .hard_loss_fn which was defined within SpliceDistillationLoss
            val_loss = distillation_loss_fn.hard_loss_fn(student_logits, hard_targets_cropped)
            total_val_loss += val_loss.item() * input_data.size(0)
            
            #Collect for metrics
            val_logits_list.append(student_logits.cpu().numpy())
            val_targets_list.append(hard_targets_cropped.cpu().numpy())

    avg_val_loss = total_val_loss / len(val_loader.dataset)
    history['val_loss'].append(avg_val_loss)


    print(f"\n--- Validation metrics (Epoch {epoch}) ---")

    #Concatenate all batches into large NumPy arrays: [N, C, L]
    #The result will be of the shape [Total_Samples, 3, L_Target]
    y_pred_concat = np.concatenate(val_logits_list, axis=0)
    y_true_concat = np.concatenate(val_targets_list, axis=0)
    
    #Evaluate Acceptor Sites (assuming channel 1)
    #Select channel 1 and flatten it to 1D: [Total_Samples * L_Target]
    y_true_acceptor = y_true_concat[:, 1, :].flatten()
    y_pred_acceptor = y_pred_concat[:, 1, :].flatten()

    print("\n\033[1m{}:\033[0m".format('Acceptor'))
   
    acceptor_metrics = print_topl_statistics(y_true_acceptor, y_pred_acceptor)

    #Evaluate Donor Sites (assuming channel 2)
    #Select channel 2 and flatten to 1D: [Total_Samples * L_Target]
    y_true_donor = y_true_concat[:, 2, :].flatten()
    y_pred_donor = y_pred_concat[:, 2, :].flatten()

    print("\n\033[1m{}:\033[0m".format('Donor'))
    donor_metrics = print_topl_statistics(y_true_donor, y_pred_donor)

    #Store both metrics in the history.
    history['val_metrics'].append({'acceptor': acceptor_metrics, 'donor': donor_metrics})
    
    
    # Checkpoint
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_student_weights = copy.deepcopy(student_model.state_dict())
        torch.save(best_student_weights, checkpoint_path)
        print(f"Checkpoint saved: Improved validation loss for {best_val_loss:.4f}")

    end_time = time.time()
    print(f"\nEpoch {epoch}. Time: {end_time - start_time:.2f}s")
    print(f"Average Training Loss: {avg_train_loss:.4f}")
    print(f"Average Validation Loss: {avg_val_loss:.4f} (Melhor: {best_val_loss:.4f})")
    print("-------------------------------------------------\n")

#Load the best weights for the final model
student_model.load_state_dict(best_student_weights)
