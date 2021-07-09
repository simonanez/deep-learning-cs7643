import math
import time
import random

# Pytorch packages
import torch
import torch.optim  as optim
import torch.nn as nn
import numpy as np
import csv
import math
import time

# Pytorch package
import torch
import torch.nn as nn
import torch.optim as optim

# Torchtest package
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

# Tqdm progress bar
from tqdm import tqdm_notebook, tqdm

# Code provide to you for training and evaluation
from utils import train, evaluate, set_seed_nb, unit_test_values
from models.naive.RNN import VanillaRNN
from models.naive.LSTM import LSTM
from models.seq2seq.Encoder import Encoder
from models.seq2seq.Decoder import Decoder
from models.seq2seq.Seq2Seq import Seq2Seq


from utils import train, evaluate, set_seed_nb, unit_test_values

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
                                          [[0.0599, -0.0151]],
                                          [[-0.9273, 0.2683]],
                                          [[0.6161, 0.5412]]])
        expected_hidden = torch.FloatTensor([[[0.4912, -0.6078],
                                              [0.4912, -0.6078],
                                              [0.4985, -0.6658],
                                              [0.4932, -0.6242],
                                              [0.4880, -0.7841]]])
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
        expected_hidden = torch.FloatTensor([[[-0.1854, 0.5561],
                                              [-0.4359, 0.1476],
                                              [-0.0992, -0.3700],
                                              [0.9429, 0.8276],
                                              [0.0372, 0.3287]]])
        return expected_out, expected_hidden

    if testcase == 'seq2seq':
        expected_out = torch.FloatTensor([[[-2.4136, -2.2861, -1.7145, -2.5612, -1.9864, -2.0557, -1.7461,
                                            -2.1898],
                                           [-2.0869, -2.9425, -2.0188, -1.6864, -2.5141, -2.3069, -1.4921,
                                            -2.3045]]])
        return expected_out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed_nb()
encoderON = False
decoderON = False
seq2seqON = False

if seq2seqON==True:
    embedding_size = 32
    hidden_size = 32
    input_size = 8
    output_size = 8
    batch, seq = 1, 2
    expected_out = unit_test_values('seq2seq') # (1, 1, 2, 8)

    encoder = Encoder(input_size, embedding_size, hidden_size, hidden_size)
    decoder = Decoder(embedding_size, hidden_size, hidden_size, output_size)

    seq2seq = Seq2Seq(encoder, decoder, 'cpu')
    x_array = np.random.rand(batch, seq) * 10
    x = torch.LongTensor(x_array)
    out = seq2seq.forward(x)

    print('Close to out: ', expected_out.allclose(out, atol=1e-4))




################# ENCODER #################
if encoderON==True:
    expected_out, expected_hidden = unit_test_values('encoder')

    i, n, h = 10, 4, 2

    encoder = Encoder(i, n, h, h)
    x_array = np.random.rand(5,1) * 10
    x = torch.LongTensor(x_array)
    out, hidden = encoder.forward(x)


    print('Close to out: ', expected_out.allclose(out, atol=1e-4))
    print('Close to hidden: ', expected_hidden.allclose(hidden, atol=1e-4))

################ DECODER #################
if decoderON==True:
    i, n, h =  10, 2, 2
    decoder = Decoder(h, n, n, i)
    x_array = np.random.rand(5, 1) * 10
    x = torch.LongTensor(x_array)
    _, enc_hidden = unit_test_values('encoder')
    out, hidden = decoder.forward(x,enc_hidden)

    expected_out, expected_hidden = unit_test_values('decoder')

    print('Close to out: ', expected_out.allclose(out, atol=1e-4))
    print('Close to hidden: ', expected_hidden.allclose(hidden, atol=1e-4))





# Define the source and target language
MAX_LEN = 20
SRC = Field(tokenize = "spacy",
            tokenizer_language="de",
            init_token = '<sos>',
            eos_token = '<eos>',
            fix_length = MAX_LEN,
            lower = True)

TRG = Field(tokenize = "spacy",
            tokenizer_language="en",
            init_token = '<sos>',
            eos_token = '<eos>',
            fix_length = MAX_LEN,
            lower = True)

# Download and split the data. It should take some time
train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'),
                                                    fields = (SRC, TRG))
# Define Batchsize
BATCH_SIZE = 128

# Build the vocabulary associated with each language
SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)

# Get the padding index to be ignored later in loss calculation
PAD_IDX = TRG.vocab.stoi['<pad>']

# Get data-loaders using BucketIterator
train_loader, valid_loader, test_loader = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = BATCH_SIZE, device = device)

# Get the input and the output sizes for model
input_size = len(SRC.vocab)
output_size = len(TRG.vocab)


# Hyperparameters. You are welcome to modify these
encoder_emb_size = 32
encoder_hidden_size = 64
encoder_dropout = 0.2

decoder_emb_size = 32
decoder_hidden_size = 64
decoder_dropout = 0.2

learning_rate = 1e-3
model_type = "RNN"

EPOCHS = 10

#input size and output size
input_size = len(SRC.vocab)
output_size = len(TRG.vocab)


# Declare models, optimizer, and loss function
encoder = Encoder(input_size, encoder_emb_size, encoder_hidden_size, decoder_hidden_size, dropout = encoder_dropout, model_type = model_type)
decoder = Decoder(decoder_emb_size, encoder_hidden_size, encoder_hidden_size, output_size, dropout = decoder_dropout, model_type = model_type)
seq2seq_model = Seq2Seq(encoder, decoder, device)
optimizer = optim.Adam(seq2seq_model.parameters(), lr = learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

for epoch_idx in range(EPOCHS):
    print("-----------------------------------")
    print("Epoch %d" % (epoch_idx + 1))
    print("-----------------------------------")

    train_loss, avg_train_loss = train(seq2seq_model, train_loader, optimizer, criterion)
    scheduler.step(train_loss)

    val_loss, avg_val_loss = evaluate(seq2seq_model, valid_loader, criterion)

    avg_train_loss = avg_train_loss.item()
    avg_val_loss = avg_val_loss.item()
    print("Training Loss: %.4f. Validation Loss: %.4f. " % (avg_train_loss, avg_val_loss))
    print("Training Perplexity: %.4f. Validation Perplexity: %.4f. " % (np.exp(avg_train_loss), np.exp(avg_val_loss)))














