import torch
import torch.nn as nn

class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes):
        '''
        :param input_dim: input feature dimension
        :param hidden_size: hidden dimension
        :param num_classes: total number of classes
        '''
        super(TwoLayerNet, self).__init__()
        #############################################################################
        # TODO: Initialize the TwoLayerNet, use sigmoid activation between layers   #
        #############################################################################
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.activation = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, num_classes)


        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        flattened_dim = x.size()[1]*x.size()[2]*x.size()[3]
        flat_x = torch.reshape(x, (len(x),flattened_dim))


        #pass data thru fully connected network 1:
        out = self.fc1(flat_x)

        #pass data thru sigmoid:
        out = self.activation(out)

        #pass data thru fully connected network 2:
        out = self.fc2(out)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out