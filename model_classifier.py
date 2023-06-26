import torch
import torch.nn as nn
import math
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GCNLayer(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        
        self.MLP_GIN = nn.Sequential(
            nn.Linear(self.in_feature, self.out_feature),
            nn.ELU()
            ).cuda()
        
    def forward(self, A, X):
        z = torch.matmul(A, X)
        z_new = self.MLP_GIN(z)
        return z_new
        

class Classifier(nn.Module):
    def __init__(self, nlayer_gcn, nnodes, nfeature_mlp, nlatent, tau):
        super(Classifier, self).__init__()
        
        self.nlayer_gcn = nlayer_gcn
        self.nnodes = nnodes
        self.nfeature_gcn = nnodes
        self.nfeature_mlp = nfeature_mlp
        self.nlatent = nlatent
        self.tau = tau
        
        # encoders
        self.gcn_fc = torch.nn.ModuleList()
        for _ in range(self.nlayer_gcn):
            self.gcn_fc.append(GCNLayer(self.nnodes, self.nnodes))
            
            
        self.gcn_sc = torch.nn.ModuleList()
        for _ in range(self.nlayer_gcn):
            self.gcn_sc.append(GCNLayer(self.nnodes, self.nnodes))
          
        # property prediction
        self.prop_pred1 = nn.Sequential(
            nn.Linear(self.nnodes * 2, self.nfeature_mlp),
            nn.ELU(),
            nn.Linear(self.nfeature_mlp, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, 16),
            nn.ELU(),
            nn.Linear(16, 4),
            )
        
        self.prop_pred2 = nn.Sequential(
            nn.Linear(self.nnodes * 2, self.nfeature_mlp),
            nn.ELU(),
            nn.Linear(self.nfeature_mlp, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, 8),
            nn.ELU(),
            nn.Linear(8, 1),
            )
        
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.nll = nn.NLLLoss()
        
  
    def forward(self, A_fc, A_sc, age, gender):
        X = torch.eye(A_sc.shape[0]).float().to(device)
        
        # encoder
        ## fc
        z_fc = X
        for layer in self.gcn_fc:
            z_fc = layer(A_fc, z_fc)
        
        ## sc
        z_sc = X
        for layer in self.gcn_sc:
            z_sc = layer(A_sc, z_sc)
        
        z_prop = torch.cat((torch.sum(z_fc, dim = 0).view(1, -1), torch.sum(z_sc, dim = 0).view(1, -1)), dim = 1)

        prop1 = torch.softmax(self.prop_pred1(z_prop).view(1, -1), dim=1)
        prop2 = torch.sigmoid(self.prop_pred2(z_prop).view(-1))


        loss = 5 * self.bce(age, prop1) + 1 * (gender - prop2) ** 2

        # print(age, prop1)
        
        return prop1, prop2, loss
    
   