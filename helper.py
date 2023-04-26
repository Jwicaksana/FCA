from __future__ import division

import cv2
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np

from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, \
 RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, RandomCrop, GaussianBlur
from torch.utils.data import Dataset, DataLoader

import os
import copy
import data_transform.transforms as extended_transforms
import data_transform.modified_randaugment as rand_augment

import torch
import torch.nn as nn
import torch.nn.functional as F 
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score, accuracy_score, \
balanced_accuracy_score, confusion_matrix, roc_curve, auc, \
precision_recall_fscore_support

colors = ['r', 'g', 'b', 'c', 'k', 'y','m', 'c','r','g','b']


#############################
# Train and test function 
#############################

def test(testloader, net, criterion, device, acc_arr=None, \
    avg_auc=None, global_=False, FCA_=False, label_binarizer=None, NUM_CLASS=0):
    net.eval()
    test_loss = 0
    probs, labels, preds =  [], [] ,[]
    with torch.no_grad():
        for batch_idx, (inputs, targets, order, one_hot) in enumerate(testloader):
            targets = targets.type(torch.LongTensor)
            inputs, targets = inputs.to(device), targets.to(device)
            if FCA_:
                y1,y2 = net(inputs)
            else:
                y1 = net(inputs)
            outputs = y1
            
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            y1 = F.softmax(y1,dim=1)
            _, predicted = y1.max(1)
            
            if FCA_ and global_:
                y2 = F.softmax(y2,dim=1)
                _, predicted = y2.max(1)
                outputs = y2
            
            predicts = predicted.cpu().detach().numpy()
            label = targets.cpu().detach().numpy()
            # Get AUC # 
            outputs = F.softmax(outputs, dim=1).cpu().detach().numpy()
            probs.extend(outputs)
            preds.extend(predicts)
            labels.extend(label)
            
    acc = balanced_accuracy_score(labels,preds)
    if acc_arr is not None: 
        acc_arr.append(acc)

    cls, _ = np.unique(labels, return_counts=True)
    labels = np.array(labels)
    probs = np.array(probs)
    '''
    compute auc per class then bauc 
    '''
    auc_ = [0.0] * NUM_CLASS
    if label_binarizer is not None:
        label_onehot = label_binarizer.transform(labels)
        # compute auc per class (1 vs many)
        fpr, tpr, roc_auc = dict(), dict(), dict()
        for i in range(NUM_CLASS):
            fpr[i], tpr[i], _ = roc_curve(label_onehot[:,i], \
                                         probs[:,i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        fpr_grid = np.linspace(0.0,1.0,1000)
        mean_tpr = np.zeros_like(fpr_grid)
        for i in range(NUM_CLASS):
            if i not in cls:
                continue
            mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])
            auc_[i] = auc(fpr_grid, np.interp(fpr_grid, fpr[i], tpr[i]))
    if avg_auc is not None:
        avg_auc.append(np.sum(np.array(auc_))/len(cls))

#####
def train(trainloader, net, optimizer, criterion, device, \
          acc_arr=None, loss_arr=None, FCA_=False, FCA_PARAMS=None):
    net.train()
    train_loss = 0
    labels, preds, probs = [], [], []
    
    kld = nn.KLDivLoss().to(device)
    for batch_idx, (inputs, targets, order, one_hot) in \
    enumerate(trainloader):
        targets = targets.type(torch.LongTensor)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        if FCA_:
            LAM1, LAM2, CONSISTENCY = FCA_PARAMS['LAM1'], FCA_PARAMS['LAM2'], FCA_PARAMS['CONSISTENCY']
            y1,y2 = net(inputs)
            l1, l2 = criterion(y1, targets), criterion(y2, targets)
            l1, l2 = torch.mean(l1), torch.mean(l2)  
            y1 = F.softmax(y1, dim=1)
            y2 = F.softmax(y2, dim=1)
            # update only the y2 since that's the head we regularize
            kloss = kld(y1+1e-11, y2.detach()+1e-11)
            loss = LAM1*l1 + LAM2*l2 + CONSISTENCY*kloss
            
            y = y2
        else:
            y1 = net(inputs)
            l1 = criterion(y1, targets)
            l1 = torch.mean(l1)
            loss = l1
            
            y = y1
        loss.backward()
        optimizer.step()
        ###############################
        outputs = y # use bsm
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        predicts = predicted.cpu().detach().numpy()
        label = targets.cpu().detach().numpy()
        '''
        adding probs
        '''
        outputs = F.softmax(outputs, dim=1).cpu().detach().numpy()
        probs.extend(outputs)
        preds.extend(predicts)
        labels.extend(label)
        
    acc = balanced_accuracy_score(labels,preds)
    if acc_arr is not None: 
        acc_arr.append(acc)
    if loss_arr is not None:
        loss_arr.append((train_loss/(batch_idx+1)))


######################################################
# Federated learning related function and helper #####
######################################################

############################################
#### federated aggregation (fedavg) 
#### input: CLIENTS <list of client>
####      : nets <collection of dictionaries>
####      : WEIGHTS of client for averaging
####      : name of federated model
####      : head name (if we need to keep them local)
############################################
def aggr_fed(CLIENTS, WEIGHTS_CL, nets, fed_name='global', head=None):
    for param_tensor in nets[fed_name].state_dict():
        if param_tensor in head:
            continue
        tmp= None
        TOTAL_CLIENTS = len(CLIENTS)
        for client, w in zip(CLIENTS, WEIGHTS_CL):
            if tmp == None:
                tmp = copy.deepcopy(w*nets[client].state_dict()[param_tensor])
            else:
                tmp += w*nets[client].state_dict()[param_tensor]
        nets[fed_name].state_dict()[param_tensor].data.copy_(tmp)
        del tmp

############################################
#### copy federated model to client 
#### input: CLIENTS <list of client>
####      : nets <collection of dictionaries>
####      : WEIGHTS of client for averaging
####      : name of federated model
####      : head name (if we need to keep them local)
############################################
def copy_fed(CLIENTS, nets, fed_name='global', head=None):
    for client in CLIENTS:
        for param_tensor in nets[fed_name].state_dict():
            if param_tensor in head:
                continue
            tmp= copy.deepcopy(nets[fed_name].state_dict()[param_tensor])
            nets[client].state_dict()[param_tensor].data.copy_(tmp)
            del tmp

###############################################
# Loss  functions                         #####
###############################################

############################################
# Focal loss
# adjust the importance of each sample
# based on the 'difficulty'
############################################
class FocalLoss(nn.CrossEntropyLoss):
    ''' Focal loss for classification tasks on imbalanced datasets '''

    def __init__(self, gamma, alpha=None, ignore_index=-100, reduction='none'):
        super().__init__(weight=alpha, ignore_index=ignore_index, reduction='none')
        self.reduction = reduction
        self.gamma = gamma

    def forward(self, input_, target):
        cross_entropy = super().forward(input_, target)
        # Temporarily mask out ignore index to '0' for valid gather-indices input.
        # This won't contribute final loss as the cross_entropy contribution
        # for these would be zero.
        target = target * (target != self.ignore_index).long()
        input_prob = torch.gather(F.softmax(input_, 1), 1, target.unsqueeze(1))
        loss = torch.pow(1 - input_prob, self.gamma) * cross_entropy
        return torch.mean(loss) if self.reduction == 'mean' else torch.sum(loss) if self.reduction == 'sum' else loss

######################################################
### Balanced softmax -> adjusted based on prior     
### parameters: device (cuda)                       
###           : balance (set true or not)           
###           : clsnm (the class ratio proportion)  
######################################################
class CustomLoss(nn.Module):
    def __init__(self, clsnm, basic_loss,device, balance=False):
        super().__init__()
        self.basic_loss = basic_loss.to(device)
        self.prior = torch.tensor(np.array(clsnm)/np.sum(clsnm)).float().to(device)
        self.balance = balance
    def forward(self, output_logits, target):
        if not self.balance:
            return self.basic_loss(output_logits, target)
        else: #balanced softmax
            output_logits = output_logits + torch.log(self.prior + 1e-9)
            return self.basic_loss(output_logits, target)

######################################
#### plot results 
#### input: num <of plot graph> 
####      : CLIENTS <str list> 
####      : index <x axis > 
####      : y_axis : value 
####      : title : legend 
######################################
def plot_graphs(num, CLIENTS, index, y_axis, title):
    idx_clr = 0
    plt.figure(num)
    for client in CLIENTS:
        plt.plot(index, y_axis[client], colors[idx_clr], label=client+ title)
        idx_clr += 1
    plt.legend()
    plt.show()

##############################################################
# function to only update the classifier or head of the model
# --> probing                                                
##############################################################
def probing(network, head, reverse=False):
    for name, param in network.named_parameters():
        gradient_requirement = reverse
        param.requires_grad=gradient_requirement
        if name in head:
            param.requires_grad=not gradient_requirement

def finetune(network):
    for name, param in network.named_parameters():
        param.requires_grad=True

def freeze(network, head, reverse=False):
    for name, param in network.named_parameters():
        if reverse:
            if name in head:
                param.requires_grad=True
            else:
                param.requires_grad=False
        else:
            if name in head:
                param.requires_grad=False
            else:
                param.requires_grad=True

class customized_effnet(nn.Module):
    def __init__(self, effnet, classnum=8):
        super(customized_effnet,self).__init__()
        self.feat= effnet.features
        self.avgpool = effnet.avgpool
        self.drop = effnet.classifier[0]
        self.standard_head = nn.Linear(in_features=1280, out_features=classnum, bias=True)
        self.standard_head2 = nn.Linear(in_features=1280, out_features=classnum, bias=True)
        
    def forward(self,x):
        x = self.feat(x)
        x = self.avgpool(x)
        x = x.flatten(start_dim=1)
        x = self.drop(x)
        
        y = self.standard_head(x)
        y2 = self.standard_head2(x)
        return y,y2             
                

class extracted_effnet_moon(nn.Module):
    def __init__(self, effnet, classnum=8):
        super(extracted_effnet_moon,self).__init__()
        self.feat= effnet.features
        self.avgpool = effnet.avgpool
        self.drop = effnet.classifier[0]
        self.projection_head = nn.Linear(in_features=1280, out_features=256, bias=True)
        self.projection_head2 = nn.Linear(in_features=256, out_features=256, bias=True)
        self.standard_head = nn.Linear(in_features=256, out_features=classnum, bias=True)
            
    def forward(self,x):
        x = self.feat(x)
        x = self.avgpool(x)
        x = x.flatten(start_dim=1)
        
        proj_feat = self.projection_head(x)
        proj_feat = F.relu(proj_feat)
        proj_feat = self.projection_head2(proj_feat)
        
        x = self.drop(proj_feat)
        y = self.standard_head(x)
        '''
        return the intermediate features
        after projection layer
        '''
        return proj_feat, y             
                
        
       
        