import torch
import torch.nn as nn

class VanillaCNN(nn.Module):
    def __init__(self):
        super(VanillaCNN, self).__init__()
        #############################################################################
        # TODO: Initialize the Vanilla CNN                                          #
        #       Conv: 7x7 kernel, stride 1 and padding 0                            #
        #       Max Pooling: 2x2 kernel, stride 2                                   #
        #############################################################################
        #convolution layer 32 output channels, kernel size 7, stride 1, zero padding.
        #kernel size of 7, with stride 1, and zero padding.
        # max pool should use kernel size of 2 and stride of 2.
        # fully connecteed should have 10 output features.
        self.conv_input_dim = 3
        self.conv_output_dim = 32
        self.convInput_padding = 0
        self.convkernel_dim = 7
        self.convkernel_stride = 1

        self.maxPoolKernel_dim    = 2
        self.maxPoolKernel_stride = 2

        self.fc_input_dim = 32*13*13 #find a way to automate this maybe?
        self.fc_output_dim = 10

        self.conv2d = nn.Conv2d(self.conv_input_dim, self.conv_output_dim, self.convkernel_dim, stride=self.convkernel_stride,
                                padding=self.convInput_padding)
        self.activation = nn.ReLU()
        self.maxpool2d = nn.MaxPool2d(self.maxPoolKernel_dim, self.maxPoolKernel_stride, padding=0)
        self.fc1 = nn.Linear(self.fc_input_dim, self.fc_output_dim)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################


    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        out = self.conv2d(x)
        out = self.activation(out)
        out = self.maxpool2d(out)

        # linear layer.
        flattened_dim = out.size()[1]*out.size()[2]*out.size()[3]
        flat_out = torch.reshape(out, (len(out),flattened_dim))
        outs = self.fc1(flat_out)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return outs