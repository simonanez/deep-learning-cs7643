import random

import torch
import torch.nn as nn
import torch.optim as optim

class Decoder(nn.Module):
    """ The Decoder module of the Seq2Seq model 
        You will need to complete the init function and the forward function.
    """
    def __init__(self, emb_size, encoder_hidden_size, decoder_hidden_size, output_size, dropout = 0.2, model_type = "RNN"):
        super(Decoder, self).__init__()

        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.output_size = output_size
        self.model_type = model_type

        #############################################################################
        # TODO:                                                                     #
        #    Initialize the following layers of the decoder in this order!:         #
        #       1) An embedding layer                                               #
        #       2) A recurrent layer, this part is controlled by the "model_type"   #
        #          argument. You need to support the following type(in string):     #
        #          "RNN", "LSTM".                                                   #
        #       3) A single linear layer with a (log)softmax layer for output       #
        #       4) A dropout layer                                                  #
        #                                                                           #
        # NOTE: Use nn.RNN and nn.LSTM instead of the naive implementation          #
        #############################################################################
        self.embedding = nn.Embedding(self.output_size,self.emb_size)
        if self.model_type == "RNN":
            self.rnn = nn.RNN(self.emb_size, self.decoder_hidden_size, batch_first=True)
        else:
            self.rnn = nn.LSTM(self.emb_size, self.decoder_hidden_size, batch_first=True)
        self.linear1 = nn.Linear(self.encoder_hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax()
        self.dropout = nn.Dropout(dropout)


        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, input, hidden):
        """ The forward pass of the decoder
            Args:
                input (tensor): the encoded sequences of shape (batch_size, 1); HINT: encoded does not mean from encoder!!
                hidden (tensor): the hidden weights of the previous time step from the decoder
            Returns:
                output (tensor): the output of the decoder
                hidden (tensor): the weights coming out of the hidden unit
        """
        

        #############################################################################
        # TODO: Implement the forward pass of the decoder.                          #
        #       Apply the dropout to the embedding layer before you apply the       #
        #       recurrent layer                                                     #
        #       Apply linear layer and softmax activation to output tensor before   #
        #       returning it.                                                       #
        #############################################################################
        input_size = input.size()
        if len(input) == 1 and len(input_size) < 2:
            input = input.unsqueeze(0)
        embedding = self.embedding(input) # input for decoder is (5,1)
        embedding = self.dropout(embedding)

        output, hidden = self.rnn(embedding, hidden)
        output = torch.squeeze(output)

        if len(input) == 1 and len(input_size) > 1:
            output = output[0, :]

        output = self.linear1(output)
        # output = torch.squeeze(output)
        output = self.softmax(output)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return output, hidden
            
