import numpy as np
import torch
import torch.nn as nn

class VanillaRNN(nn.Module):
    """ An implementation of vanilla RNN using Pytorch Linear layers and activations.
        You will need to complete the class init function, forward function and hidden layer initialization.
    """

    def __init__(self, input_size, hidden_size, output_size):
        """ Init function for VanillaRNN class
            Args:
                input_size (int): the number of features in the inputs.
                hidden_size (int): the size of the hidden layer
                output_size (int): the size of the output layer

            Returns: 
                None
        """
        super(VanillaRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        #############################################################################
        # TODO:                                                                     #
        #    Initialize parameters and layers. You should                           #
        #    include a hidden unit, an output unit, a tanh function for the hidden  #
        #    unit, and a log softmax for the output unit.                           #
        #    You MUST NOT use Pytorch RNN layers(nn.RNN, nn.LSTM, etc).             #
        #    Initialize the hidden layer before the output layer!                   #
        #############################################################################

        self.linear1 = nn.Linear( self.input_size + self.hidden_size, self.hidden_size)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear( self.input_size + self.hidden_size, self.output_size )
        self.softmax = nn.LogSoftmax(dim=1)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, input, hidden):
        """ The forward function of the Vanilla RNN
            Args:
                input (tensor): a batch of data of shape (batch_size, input_size) at one time step
                hidden (tensor): the hidden value of previous time step of shape (batch_size, hidden_size)

            Returns:
                output (tensor): the output tensor of shape (batch_size, output_size)
                hidden (tensor): the hidden value of current time step of shape (batch_size, hidden_size)
        """


        #############################################################################
        # TODO:                                                                     #
        #   Implement the forward pass for the Vanilla RNN. Note that we are only   #
        #   going over one time step. Please refer to the structure in the notebook.#                                              #
        #############################################################################
        concat = torch.cat((input,hidden), 1 )
        hidden = self.linear1(concat)
        hidden = self.tanh(hidden)

        output = self.linear2(concat)
        output = self.softmax(output)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return output, hidden

