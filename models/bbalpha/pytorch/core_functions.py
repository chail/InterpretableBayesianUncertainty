from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F



class MNISTDataset(Dataset):
    def __init__(self, images, labels, reshape=False, transforms=None):
        self.images = np.reshape(images, (-1, 1, 28, 28)) if reshape else images
        self.labels = labels # one hot labels
        self.transforms = transforms
        
    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]
        if self.transforms is not None:
            img = self.transforms(img)
        return (img, label)

    def __len__(self):
        return self.images.shape[0]
    
def to_categorical_mc(onehot, reps):
    """ 1-hot with MC replicates, encodes a tensor """
    onehot_mc = torch.squeeze(torch.cat(reps * [onehot[:, None]], dim=1))
    return onehot_mc.float()

def logsumexp(x, axis):
    x_max, _ = torch.max(x, axis, keepdim=True)
    return torch.log(torch.sum(torch.exp(x - x_max), axis, keepdim=True)) + x_max

def bbalpha_softmax_cross_entropy_with_mc_logits(alpha):
    if alpha != 0.0:
        # alpha = torch.tensor(alpha)
        def bbalpha_loss(mc_logits, targets):
            # log(p_ij), p_ij = softmax(logit_ij)
            #assert mc_logits.ndim == 3
            mc_log_softmax = mc_logits - torch.max(mc_logits, dim=2, keepdim=True)[0]
            mc_log_softmax = mc_log_softmax - torch.log(torch.sum(torch.exp(mc_log_softmax), dim=2, keepdim=True))
            mc_ll = torch.sum(targets * mc_log_softmax, dim=2)  # N x K
            # this is the loss function (note inside is also multiplied by alpha
            K_mc = mc_ll.shape[1]
            return - 1. / alpha * torch.mean((logsumexp(alpha * mc_ll, 1) + torch.log(torch.tensor(1.0 / K_mc)))) 
    else:
        criterion = nn.CrossEntropyLoss()
        def bbalpha_loss(mc_logits, targets):
            K = mc_logits.size(1)
            # cross entropy loss for each MC sample
            labels = torch.argmax(targets[:, 0, :])
            mc_ce = torch.stack([criterion(mc_outputs[:, ii, :], labels) for ii in range(K)]) 
            return torch.mean(mc_ce) # return mean over MC samples
    return bbalpha_loss

# evaluation for classification tasks
def test_MC_dropout(mc_logits, targets):
    mc_logits = mc_logits
    mc_log_softmax = mc_logits - torch.max(mc_logits, dim=2, keepdim=True)[0]
    mc_log_softmax = mc_log_softmax - torch.log(torch.sum(torch.exp(mc_log_softmax), dim=2, keepdim=True))
    mc_softmax = torch.exp(mc_log_softmax)
    pred = torch.mean(mc_softmax, dim=1) # average over MC samples
    acc = torch.mean(torch.argmax(pred, dim=1).eq(torch.argmax(targets, dim=1)).float())
    ll = torch.mean(torch.log(torch.sum(pred * targets.float(), dim=1))) # sum over D classes and N samples
    return acc, ll

# raw accuracy without averaging over MC samples
def test_acc(mc_logits, targets_mc):
    mc_logits = mc_logits
    mc_log_softmax = mc_logits - torch.max(mc_logits, dim=2, keepdim=True)[0]
    mc_log_softmax = mc_log_softmax - torch.log(torch.sum(torch.exp(mc_log_softmax), dim=2, keepdim=True))
    mc_softmax = torch.exp(mc_log_softmax)
    acc = torch.mean(torch.argmax(mc_log_softmax, dim=-1).eq(torch.argmax(targets_mc, dim=-1)).float())
    return acc


# define MLP with 2 hidden layers
class Net(nn.Module):

    def __init__(self, input_dim, nb_units, nb_classes, dropout):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(input_dim, nb_units)
        self.fc2 = nn.Linear(nb_units, nb_units)
        self.fc3 = nn.Linear(nb_units, nb_classes) 
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
    
# CNN model
class CNNNet(nn.Module):

    def __init__(self, input_size, in_channels, 
                 nb_units, nb_classes, dropout):
        super(CNNNet, self).__init__()
        nb_filters = 32
        pool_size = 2
        kernel_size = 3
        
        self.conv1 = nn.Conv2d(in_channels, nb_filters, 
                               kernel_size, padding=0)
        self.conv2 = nn.Conv2d(nb_filters, nb_filters,
                               kernel_size, padding=0)
        self.pool = nn.MaxPool2d(pool_size)
        self.flatten_dim = int(nb_filters * 
                               ((input_size - 4) / 2)**2)
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(self.flatten_dim, nb_units)
        self.fc2 = nn.Linear(nb_units, nb_classes) 

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.flatten_dim)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

