# -*- coding: utf-8 -*-
import torch
import numpy as np
import pandas as pd
from model import *
from model_classifier import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
# Load data
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

# load the trained classification model
classifier_model = Classifier(nlayer_gcn = 2, nnodes = scs[0, :, :].shape[0], nfeature_mlp = 256, nlatent = 256, tau = 10).to(device)
classifier_model.load_state_dict(torch.load('saved_classification_model.pt')['model_state_dict'])
classifier_model.eval()

# Freeze the parameters
for param in classifier_model.parameters():
    param.requires_grad = False

# Sirui, you should load a pre-trained MolGen model with Shiyu's training code here
model_path = 'model_0.pt'
# Load the saved model
checkpoint = torch.load(model_path)

model = MolGen(nlayer_gcn = 2, nnodes = scs[0, :, :].shape[0], nfeature_mlp = 256, nlatent = 256, tau = 10)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)

# Load the saved state dictionaries into the model
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)


def accuracy(pred,label):
    correct_predictions = 0
    total_examples = len(pred)
    for pred_label, gold_label in zip(pred, label):
        if pred_label == gold_label:
            correct_predictions += 1
    return correct_predictions / total_examples

train_idx, valid_idx = train_test_split(np.arange(len(ids)), test_size=0.8, random_state=42)
train_ids, valid_ids = ids[train_idx], ids[valid_idx]
train_fcs, valid_fcs = fcs[train_idx], fcs[valid_idx]
train_scs, valid_scs = scs[train_idx], scs[valid_idx]
train_ages, valid_ages = [ages_tensor[i] for i in train_idx], [ages_tensor[i] for i in valid_idx]
train_gender, valid_gender = gender[train_idx], gender[valid_idx]


# model = MolGen(nlayer_gcn = 2, nnodes = scs[0, :, :].shape[0], nfeature_mlp = 256, nlatent = 256, tau = 10).to(device)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

epochs = 30
epoch = 0

# For balancing seen and generated data
n_iter_1 = 60
n_iter_2 = 30

val_accs1 = []
val_accs2 = []
    
for epoch in range(epochs):
    model.train()
    
    total_loss_1 = 0
    total_loss_2 = 0
    preds1 = []
    preds2 = []
    ages_gold = []
    gender_gold = []

    # Train with generated data
    for _ in range(n_iter_1):
        A_fc_gen, A_sc_gen, prop1_pred, prop2_pred = model.sample()
        prop1_true, prop2_true, loss_prop = classifier_model(A_fc_gen, A_sc_gen, prop1_pred, prop2_pred)
        # total_loss_1 += loss_prop
        loss_graph_rec, loss_prop_rec, loss_kl_disentangle, loss_kl, prop1, prop2, mask1, mask2 = model(A_fc = A_fc_gen, A_sc = A_sc_gen, age = prop1_true, gender = prop2_true)
        loss_norm = torch.sum(mask1) + torch.sum(mask2)
        loss_1 = loss_graph_rec + loss_prop_rec + loss_kl+ loss_kl_disentangle + loss_norm
        total_loss_1 += loss_1
    
    # Train with data in training set
    for _ in range(n_iter_2):
        i = torch.randint(0, len(train_ids), (1,)).item()
        loss_graph_rec, loss_prop_rec, loss_kl_disentangle, loss_kl, prop1, prop2, mask1, mask2 = model(A_fc = train_fcs[i, :, :].float().to(device), A_sc = train_scs[i, :, :].float().to(device), age = train_ages[i].float().to(device), gender = train_gender[i].view(-1).float().to(device))
        loss_norm = torch.sum(mask1) + torch.sum(mask2)
        loss_2 = loss_graph_rec + loss_prop_rec + loss_kl+ loss_kl_disentangle + loss_norm
        total_loss_2 += loss_2
        
        # age prediction
        _, pred1 = torch.max(prop1.data, 1)
        preds1.append(pred1)
        
        # gender prediction
        preds2.append((prop2 > 0.5).item())
        
    
    total_loss =  total_loss_1 + total_loss_2
    
    
    total_loss.backward()
    optimizer.step()
    

    model.eval()
    with torch.no_grad():
        val_preds1 = []
        val_preds2 = []
        for i in valid_idx:
            optimizer.zero_grad()
            loss_graph_rec, loss_prop_rec, loss_kl_disentangle, loss_kl, prop1, prop2, mask1, mask2 = model(A_fc = fcs[i, :, :].float().to(device), A_sc = scs[i, :, :].float().to(device), age = ages_tensor[i].float().to(device), gender = gender[i].view(-1).float().to(device))
            # age prediction
            _, pred1 = torch.max(prop1.data, 1)
            val_preds1.append(pred1)
            # gender prediction
            pred2 = (prop2 > 0.5).float()
            val_preds2.append(pred2)


        val_acc1 = accuracy(val_preds1,[ages[i] for i in valid_idx])
        val_accs1.append(val_acc1)
        val_acc2 = accuracy(val_preds2,[gender.tolist()[i] for i in valid_idx])
        val_accs2.append(val_acc2)
    
    # print(val_preds2)

    print(f'Train epoch: {epoch} val_acc1:{val_acc1} val_acc2:{val_acc2} loss: {total_loss/(n_iter_1+n_iter_2)}')
        
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, "model_iter.pt")


# Create x-axis values (assuming one accuracy value per data point)
x = range(1, len(val_accs1) + 1)

# Plotting the accuracies
plt.plot(x, val_accs1, marker='o')


x = range(1, len(val_accs2) + 1)

# Plotting the accuracies
plt.plot(x, val_accs2, marker='o')

# Save the plot as a PNG file
plt.savefig('accuracy1_plot.png')

# Display the plot
plt.show()