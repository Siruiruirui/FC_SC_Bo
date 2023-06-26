import torch
import numpy as np
import pandas as pd
from model_classifier import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = np.load('SC_FC_demos.npz')

ages =data["Ages"]
ages = list(pd.factorize(ages)[0])
ages_tensor = [torch.zeros(4).view(1, -1) for i in range(len(ages))]
for i in range(len(ages)):
    ages_tensor[i][0, ages[i]] = 1

fcs = torch.from_numpy(data["FCs"])
gender = data["Gender"]
gender = pd.factorize(gender)[0]
gender = torch.from_numpy(gender)
ids = torch.from_numpy(data["ids"])
scs = torch.from_numpy(data["SCs"])

train_idx, valid_idx = train_test_split(np.arange(len(ids)), test_size=0.2, random_state=42)

train_ids, valid_ids = ids[train_idx], ids[valid_idx]
train_fcs, valid_fcs = fcs[train_idx], fcs[valid_idx]
train_scs, valid_scs = scs[train_idx], scs[valid_idx]
train_ages, valid_ages = [ages_tensor[i] for i in train_idx], [ages_tensor[i] for i in valid_idx]
train_gender, valid_gender = gender[train_idx], gender[valid_idx]


model = Classifier(nlayer_gcn = 2, nnodes = scs[0, :, :].shape[0], nfeature_mlp = 256, nlatent = 256, tau = 10).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

epochs = 25
epoch = 0
i = 0



for epoch in range(epochs):
    # TRAIN
    model.train()
    total_loss = 0
    for i in range(len(train_ids)):
        optimizer.zero_grad()
        prop1, prop2, loss = model(A_fc = train_fcs[i].float().to(device), A_sc = train_scs[i].float().to(device), age = train_ages[i].float().to(device), gender = train_gender[i].view(-1).float().to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f'Train epoch: {epoch} loss: {total_loss/len(train_ids)}')

    # VALID
    model.eval()
    with torch.no_grad():
        total_valid_loss = 0
        prop1_preds, prop2_preds = [], []
        prop1_true, prop2_true = [], []
        for i in range(len(valid_ids)):
            prop1, prop2, loss = model(A_fc = valid_fcs[i].float().to(device), A_sc = valid_scs[i].float().to(device), age = valid_ages[i].float().to(device), gender = valid_gender[i].view(-1).float().to(device))
            total_valid_loss += loss.item()
            
            # Convert outputs to predicted classes
            _, prop1_pred = torch.max(prop1, dim=1)
            prop1_preds.append(prop1_pred.cpu().numpy())
            # print(prop1, prop1_pred, np.argmax(prop1_pred.cpu().numpy()))
            prop2_preds.extend((prop2>0.5).float().cpu().numpy())  # Assuming prop2 is a probability and you are using a threshold of 0.5
            
            # Add true labels
            prop1_true.append(np.argmax(valid_ages[i].cpu().numpy()))
            prop2_true.extend([valid_gender[i].cpu().numpy()])
        
        # Compute accuracy, precision, recall and F1-score
        prop1_accuracy = accuracy_score(prop1_true, prop1_preds)
        prop1_precision = precision_score(prop1_true, prop1_preds, average='weighted', zero_division=1)
        prop1_recall = recall_score(prop1_true, prop1_preds, average='weighted')
        prop1_f1 = f1_score(prop1_true, prop1_preds, average='weighted')
        
        prop2_accuracy = accuracy_score(prop2_true, prop2_preds)
        prop2_precision = precision_score(prop2_true, prop2_preds, average='binary', zero_division=1)  # Assuming binary classification for prop2
        prop2_recall = recall_score(prop2_true, prop2_preds, average='binary')
        prop2_f1 = f1_score(prop2_true, prop2_preds, average='binary')
        
    print(f'Valid epoch: {epoch}, loss: {total_valid_loss/len(valid_ids)}, Prop1 - accuracy: {prop1_accuracy}, precision: {prop1_precision}, recall: {prop1_recall}, F1: {prop1_f1}, Prop2 - accuracy: {prop2_accuracy}, precision: {prop2_precision}, recall: {prop2_recall}, F1: {prop2_f1}')
     
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, "classification_model.pt")
