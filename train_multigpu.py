import argparse
import os
import re

import monai
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from torch.optim import lr_scheduler
from tqdm import tqdm
#from config import model
import config
from dataset import BrainRSNADataset

parser = argparse.ArgumentParser()
parser.add_argument("--fold", default=0, type=int)
parser.add_argument("--type", default="FLAIR", type=str)
parser.add_argument("--model_name", default="b0", type=str)
args = parser.parse_args()


############Function definition starts##################
def gen_cm(preds,true_labels,auc_score_adj_best = 0):
        preds = np.vstack(preds).T[0].tolist()
        true_labels = np.hstack(true_labels).tolist()
        #case_ids = np.hstack(case_ids).tolist()
        auc_score = roc_auc_score(true_labels, preds)
        #auc_score_adj_best = 0
        if auc_score_adj_best==0:
            for thresh in np.linspace(0, 1, 50):
                auc_score_adj = roc_auc_score(true_labels, list(np.array(preds) > thresh))
                if auc_score_adj > auc_score_adj_best:
                    best_thresh = thresh
                    auc_score_adj_best = auc_score_adj
        else:
            best_thresh=auc_score_adj_best
            auc_score_adj_best = roc_auc_score(true_labels, list(np.array(preds) > best_thresh))

        cm = confusion_matrix(true_labels,list(np.array(preds) > best_thresh))
        return cm, best_thresh,auc_score_adj_best

def eval_cm(cm1):
        total1=sum(sum(cm1))
        #####from confusion matrix calculate accuracy
        acc=(cm1[0,0]+cm1[1,1])/total1
        #print ('Accuracy : ', accuracy1)

        sens = cm1[0,0]/(cm1[0,0]+cm1[0,1])
        #print('Sensitivity : ', sensitivity1 )

        spec = cm1[1,1]/(cm1[1,0]+cm1[1,1])

        return acc,sens,spec
############Function definition end##################

#data = pd.read_csv("../Input/train.csv")
data = pd.read_csv(config.label_file)
train_df = data[data.fold != args.fold].reset_index(drop=False)
val_df = data[data.fold == args.fold].reset_index(drop=False)


device = torch.device("cuda:1")

print(f"train_{args.type}_{args.fold}")
train_dataset = BrainRSNADataset(data=train_df, mri_type=args.type, ds_type=f"train_{args.type}_{args.fold}")

valid_dataset = BrainRSNADataset(data=val_df, mri_type=args.type, ds_type=f"val_{args.type}_{args.fold}")


train_dl = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=config.TRAINING_BATCH_SIZE,
    shuffle=True,
    num_workers=config.n_workers,
    drop_last=True,
    pin_memory=True,
)

validation_dl = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=config.TEST_BATCH_SIZE,
    shuffle=False,
    num_workers=config.n_workers,
    pin_memory=True,
)


#model = monai.networks.nets.resnet101(spatial_dims=3, n_input_channels=1, n_classes=1)
#model = monai.networks.nets.resnet10(spatial_dims=3, n_input_channels=1, n_classes=1)
#model = monai.networks.nets.resnet50(spatial_dims=3, n_input_channels=1, n_classes=1)
#model = monai.networks.nets.EfficientNetBN("efficientnet-b0",spatial_dims=3, in_channels=1, num_classes=1)
model =config.MODEL
optimizer = optim.Adam(model.parameters(), lr=0.0001)
#loss_function = torch.nn.CrossEntropyLoss()

scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.5, last_epoch=-1, verbose=True)

model.zero_grad()
#model = nn.DataParallel(model)
#model = nn.DataParallel(model, device_ids=[0, 1, 2]).cuda()
#model = nn.DataParallel(model, device_ids=[0]).cuda()
device = torch.device("cuda:1") ## specify the GPU id's, GPU id's start from 0.
model= nn.DataParallel(model,device_ids = [1, 2])
model.to(device)
best_loss = 9999
best_auc = 0
criterion = nn.BCEWithLogitsLoss()
fname=f"{args.model_name}_epoch_{config.N_EPOCHS}.txt"
#myfile = open(fname, 'w')
#Result_df=pd.DataFrame(columns=['Epoch','Batch','auc_score','Accuracy','Sens','Spec'])
#Train_Result_df=pd.DataFrame(columns=['Epoch','Batch','auc_score','Accuracy','Sens','Spec'])
Results=[]
Train_Results=[]
for counter in range(config.N_EPOCHS):

    epoch_iterator_train = tqdm(train_dl)
    tr_loss = 0.0
    train_preds = []
    train_labels = []
    train_case_ids = []
    for step, batch in enumerate(epoch_iterator_train):
        model.train()
        images, targets = batch["image"].to(device), batch["target"].to(device)

        outputs = model(images)
        targets = targets  # .view(-1, 1)
        loss = criterion(outputs.squeeze(1), targets.float())

        loss.backward()
        optimizer.step()
        model.zero_grad()
        optimizer.zero_grad()

        tr_loss += loss.item()
        epoch_iterator_train.set_postfix(
            batch_loss=(loss.item()), loss=(tr_loss / (step + 1))
        )
        train_preds.append(outputs.sigmoid().detach().cpu().numpy())
        train_labels.append(targets.cpu().numpy())
        train_case_ids.append(batch["case_id"])
    cm_train, best_thresh,auc_score_train=gen_cm(train_preds,train_labels)
    acc_train,sens,spec=eval_cm(cm_train)
    #Train_Result_df=Result_df.append({'Epoch':counter,'Batch':step+1,'auc_score':auc_score_train,'Accuracy':acc_train,'Sens':sens,'Spec':spec},ignore_index=True)
    tr_entry={'Epoch':counter,'Batch':step+1,'auc_score':auc_score_train,'Accuracy':acc_train,'Sens':sens,'Spec':spec}
    Train_Results.append(tr_entry)



    scheduler.step()  # Update learning rate schedule

    if config.do_valid:
        with torch.no_grad():
            val_loss = 0.0
            preds = []
            true_labels = []
            case_ids = []
            epoch_iterator_val = tqdm(validation_dl)
            for step, batch in enumerate(epoch_iterator_val):
                model.eval()
                images, targets = batch["image"].to(device), batch["target"].to(device)

                outputs = model(images)
                targets = targets  # .view(-1, 1)
                loss = criterion(outputs.squeeze(1), targets.float())
                val_loss += loss.item()
                epoch_iterator_val.set_postfix(
                    batch_loss=(loss.item()), loss=(val_loss / (step + 1))
                )
                preds.append(outputs.sigmoid().detach().cpu().numpy())
                true_labels.append(targets.cpu().numpy())
                case_ids.append(batch["case_id"])
        preds = np.vstack(preds).T[0].tolist()
        true_labels = np.hstack(true_labels).tolist()
        case_ids = np.hstack(case_ids).tolist()
        auc_score = roc_auc_score(true_labels, preds)
        auc_score_adj_best = 0
        for thresh in np.linspace(0, 1, 50):
            auc_score_adj = roc_auc_score(true_labels, list(np.array(preds) > thresh))
            if auc_score_adj > auc_score_adj_best:
                best_thresh = thresh
                auc_score_adj_best = auc_score_adj

        cm1 = confusion_matrix(true_labels,list(np.array(preds) > best_thresh))
        cm1,best_thres,auc_score =gen_cm(preds,true_labels,best_thresh)
        acc,sens,spec=eval_cm(cm1)
        

        total1=sum(sum(cm1))
        #####from confusion matrix calculate accuracy
        acc=(cm1[0,0]+cm1[1,1])/total1
        #print ('Accuracy : ', accuracy1)

        sens = cm1[0,0]/(cm1[0,0]+cm1[0,1])
        #print('Sensitivity : ', sensitivity1 )

        spec = cm1[1,1]/(cm1[1,0]+cm1[1,1])
        #print('Specificity : ', specificity1)
        #msg=f"EPOCH {counter}/{config.N_EPOCHS}: Validation average loss: {val_loss/(step+1)} + AUC SCORE = {auc_score} + AUC SCORE THRESH {best_thresh} = {auc_score_adj_best}"
        msg=f"EPOCH {counter}/{config.N_EPOCHS}: Batch: {step+1} + AUC SCORE = {auc_score_train} + ACCURACY {acc_train} test-> AUC SCORE = {auc_score} + ACCURACY {acc}"
        #Result_df=Result_df.append({'Epoch':counter,'Batch':step+1,'auc_score':auc_score,'Accuracy':acc,'Sens':sens,'Spec':spec},ignore_index=True)
        entry={'Epoch':counter,'Batch':step+1,'auc_score':auc_score,'Accuracy':acc,'Sens':sens,'Spec':spec}
        Results.append(entry)
        #myfile.writelines(msg)
        print(msg)
        if auc_score > best_auc:
            print("Saving the model...")

            
            all_files = os.listdir(config.out_dir+"weights")

            for f in all_files:
                my_pattern=f"{args.model_name}_{args.type}_fold{args.fold}_\d*\.?\d+_epochs_{config.N_EPOCHS}"
                if re.search(my_pattern,f):
                #if f"{args.model_name}_{args.type}_fold{args.fold}_epoch_{config.N_EPOCHS}" in f:
                    
                    os.remove(f"{config.out_dir}weights/{f}")

            best_auc = auc_score
            torch.save(
                model.state_dict(),
                f"{config.out_dir}weights/3d-{args.model_name}_{args.type}_fold{args.fold}_{round(best_auc,3)}_epochs_{config.N_EPOCHS}.pth",
            )

#myfile.close()
res_f=f'{config.out_dir}Test_3d--{args.model_name}_{args.type}_fold{args.fold}_epoch_{config.N_EPOCHS}.csv'
Result_df=pd.DataFrame(Results)
Result_df.to_csv(res_f,index=False)
res_f=f'{config.out_dir}Train_3d--{args.model_name}_{args.type}_fold{args.fold}_epoch_{config.N_EPOCHS}.csv'
Train_Result_df=pd.DataFrame(Train_Results)
Train_Result_df.to_csv(res_f,index=False)
#print(best_auc)
