import random

import torch
import torch.nn as nn
import torch.optim as optim

class Encoder(nn.Module):
    """ The Encoder module of the Seq2Seq model 
        You will need to complete the init function and the forward function.
    """
    def __init__(self, input_size, emb_size, encoder_hidden_size, decoder_hidden_size, dropout = 0.2, model_type = "RNN"):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.model_type = model_type
        #############################################################################
        # TODO:                                                                     #
        #    Initialize the following layers of the encoder in this order!:         #
        #       1) An embedding layer                                               #
        #       2) A recurrent layer, this part is controlled by the "model_type"   #
        #          argument. You need to support the following type(in string):     #
        #          "RNN" and "LSTM".                                                #
        #       3) Linear layers with ReLU activation in between to get the         #
        #          hidden weights of the Encoder(namely, Linear - ReLU - Linear).   #
        #          The size of the output of the first linear layer is the same as  #
        #          its input size.                                                  #
        #          HINT: the size of the output of the second linear layer must     #
        #          satisfy certain constraint relevant to the decoder.              #
        #       4) A dropout layer                                                  #
        #                                                                           #
        # NOTE: Use nn.RNN and nn.LSTM instead of the naive implementation          #
        #############################################################################
        self.embedding = nn.Embedding(self.input_size,self.emb_size)
        if self.model_type == "RNN":
            self.rnn = nn.RNN(self.emb_size, self.encoder_hidden_size, batch_first=True)
        else:
            self.rnn = nn.LSTM(self.emb_size, self.encoder_hidden_size, batch_first=True)

        self.linear1 = nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(self.encoder_hidden_size, self.decoder_hidden_size)
        self.activation_final = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, input):
        """ The forward pass of the encoder
            Args:
                input (tensor): the encoded sequences of shape (batch_size, seq_len)

            Returns:
                output (tensor): the output of the Encoder; later fed into the Decoder.
                hidden (tensor): the weights coming out of the last hidden unit
        """

        
        #############################################################################
        # TODO: Implement the forward pass of the encoder.                          #
        #       Apply the dropout to the embedding layer before you apply the       #
        #       recurrent layer                                                     #
        #       Apply tanh activation to the hidden tensor before returning it      #
        #############################################################################

        # input is (batch size, seq_len)
        # output is (batch size, seq len, embedding dim)
        embedding = self.embedding(input)
        # dropout certain signals. still output is (batch_size, seq len , embedding dim)
        embedding = self.dropout(embedding)
        # batch size dim is first.
        output, hidden = self.rnn(embedding)

        # pass in hidden thru linear layers.
        hidden = self.linear1(hidden)
        hidden = self.activation(hidden)
        hidden = self.linear2(hidden)
        # apply final tanh activation
        hidden = self.activation_final(hidden)

        # hidden state will be used for decoder's first step.

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return output, hidden