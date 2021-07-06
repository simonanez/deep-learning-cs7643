import math
import time
import random

# Pytorch packages
import torch
import torch.optim  as optim
import torch.nn as nn

# Numpy
import numpy as np

# Tqdm progress bar
from tqdm import tqdm_notebook

RANDOM_SEED = 0
def set_seed():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

def set_seed_nb():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED + 1)

def train(model, dataloader, optimizer, criterion, scheduler = None):

    model.train()

    # Record total loss
    total_loss = 0.

    # Get the progress bar for later modification
    progress_bar = tqdm_notebook(dataloader, ascii = True)

    # Mini-batch training
    for batch_idx, data in enumerate(progress_bar):
        
        source = data.src.transpose(1,0)
        target = data.trg.transpose(1,0)

        translation = model(source)
        translation = translation.reshape(-1, translation.shape[-1])
        target = target.reshape(-1)

        optimizer.zero_grad()
        loss = criterion(translation, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss
        progress_bar.set_description_str("Batch: %d, Loss: %.4f" % ((batch_idx+1), loss.item()))
    
    return total_loss, total_loss / len(dataloader)

def evaluate(model, dataloader, criterion):

    # Set the model to eval mode to avoid weights update
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        # Get the progress bar 
        progress_bar = tqdm_notebook(dataloader, ascii = True)
        for batch_idx, data in enumerate(progress_bar):
            source = data.src.transpose(1,0)
            target = data.trg.transpose(1,0)

            translation = model(source)
            translation = translation.reshape(-1, translation.shape[-1])
            target = target.reshape(-1)

            loss = criterion(translation, target)
            total_loss += loss
            progress_bar.set_description_str("Batch: %d, Loss: %.4f" % ((batch_idx+1), loss.item()))
    
    avg_loss = total_loss / len(dataloader)
    return total_loss, avg_loss

def unit_test_values(testcase):
    if testcase == 'rnn':
        return torch.FloatTensor([[-0.9827, -0.5264, -3.3529],
         [-0.9810, -0.5418, -3.1382],
         [-0.9813, -0.5594, -2.9257],
         [-0.9843, -0.5795, -2.7158]]), torch.FloatTensor([[ 0.7531,  0.8514,  0.0764,  0.3671],
         [ 0.4500,  0.7670,  0.2058,  0.0314],
         [-0.0109,  0.6440,  0.3284, -0.3116],
         [-0.4671,  0.4752,  0.4408, -0.5889]])
    
    if testcase == 'lstm':
        ht = torch.FloatTensor([[-0.0325,  0.1519,  0.0063,  0.3199],
        [ 0.1009,  0.0199,  0.2986, -0.2799],
        [ 0.1365,  0.0128,  0.3263, -0.1227],
        [ 0.1156,  0.0043,  0.2449, -0.0333]])
        ct = torch.FloatTensor([[-0.0379,  0.3315,  0.0066,  0.5134],
        [ 0.1333,  0.1196,  0.3492, -0.9668],
        [ 0.2017,  0.2715,  0.4971, -2.4863],
        [ 0.1979,  0.3571,  0.6673, -2.8806]])

        return ht, ct

    if testcase == 'encoder':
        expected_out =torch.FloatTensor([[[-0.7773, -0.2031]],
        [[-0.4129, -0.1802]],
        [[ 0.0599, -0.0151]],
        [[-0.9273,  0.2683]],
        [[ 0.6161,  0.5412]]])
        expected_hidden = torch.FloatTensor([[[ 0.4912, -0.6078],
         [ 0.4912, -0.6078],
         [ 0.4985, -0.6658],
         [ 0.4932, -0.6242],
         [ 0.4880, -0.7841]]])
        return expected_out, expected_hidden
    
    if testcase == 'decoder':
        expected_out = torch.FloatTensor([[-2.1507, -1.6473, -3.1772, -3.2119, -2.6847, -2.1598, -1.9192, -1.8130,
         -2.6142, -3.1621],
        [-2.0260, -2.0121, -3.2508, -3.1249, -2.4581, -1.8520, -2.0798, -1.7596,
         -2.6393, -3.2001],
        [-2.1078, -2.2130, -3.1951, -2.7392, -2.1194, -1.8174, -2.1087, -2.0006,
         -2.4518, -3.2652],
        [-2.7016, -1.1364, -3.0247, -2.9801, -2.8750, -3.0020, -1.6711, -2.4177,
         -2.3906, -3.2773],
        [-2.2018, -1.6935, -3.1234, -2.9987, -2.5178, -2.1728, -1.8997, -1.9418,
         -2.4945, -3.1804]])
        expected_hidden = torch.FloatTensor([[[-0.1854,  0.5561],
         [-0.4359,  0.1476],
         [-0.0992, -0.3700],
         [ 0.9429,  0.8276],
         [ 0.0372,  0.3287]]])
        return expected_out, expected_hidden

    if testcase == 'seq2seq':
        expected_out = torch.FloatTensor([[[-2.4136, -2.2861, -1.7145, -2.5612, -1.9864, -2.0557, -1.7461,
          -2.1898],
         [-2.0869, -2.9425, -2.0188, -1.6864, -2.5141, -2.3069, -1.4921,
          -2.3045]]])
        return expected_out