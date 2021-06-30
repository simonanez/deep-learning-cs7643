import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        #############################################################################
        # TODO: Initialize the network weights                                      #
        #############################################################################

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.convkernel_stride = 1
        self.convInput_padding = 1
        self.convkernel_dim = 3
        self.conv1_input_dim = 3
        self.conv1_output_dim = 30
        self.batch_norm1_output = 30

        self.conv2_input_dim = 30
        self.conv2_output_dim = 27

        self.conv3_input_dim = 27
        self.conv3_output_dim = 24

        self.conv4_input_dim = 24
        self.conv4_output_dim = 20

        self.conv5_input_dim = 20
        self.conv5_output_dim = 15

        self.conv6_input_dim = 15
        self.conv6_output_dim = 10

        self.fc1_input_dim = 10*32*32
        self.fc1_output_dim = 1000
        self.fc2_input_dim = 1000
        self.fc2_output_dim = 600
        self.fc3_input_dim = 600
        self.fc3_output_dim = 300
        self.fc4_input_dim = 300
        self.fc4_output_dim = 10

        self.conv2d_1 = nn.Conv2d(self.conv1_input_dim, self.conv1_output_dim, self.convkernel_dim,
                            stride=self.convkernel_stride,
                            padding=self.convInput_padding)
        self.activation  = nn.ReLU()
        self.batch2d_norm1 = nn.BatchNorm2d(self.conv1_output_dim)
        self.conv2d_2 = nn.Conv2d(self.conv2_input_dim, self.conv2_output_dim, self.convkernel_dim,
                                  stride=self.convkernel_stride,
                                  padding=self.convInput_padding)
        self.conv2d_3 = nn.Conv2d(self.conv3_input_dim, self.conv3_output_dim, self.convkernel_dim,
                                  stride=self.convkernel_stride,
                                  padding=self.convInput_padding)
        self.batch2d_norm2 = nn.BatchNorm2d(self.conv3_output_dim)

        self.conv2d_4 = nn.Conv2d(self.conv4_input_dim, self.conv4_output_dim, self.convkernel_dim,
                                  stride=self.convkernel_stride,
                                  padding=self.convInput_padding)
        self.conv2d_5 = nn.Conv2d(self.conv5_input_dim, self.conv5_output_dim, self.convkernel_dim,
                                  stride=self.convkernel_stride,
                                  padding=self.convInput_padding)
        self.conv2d_6 = nn.Conv2d(self.conv6_input_dim, self.conv6_output_dim, self.convkernel_dim,
                                  stride=self.convkernel_stride,
                                  padding=self.convInput_padding)
        self.resconv2d_1 = nn.Conv2d(self.conv1_output_dim, self.conv3_output_dim, self.convkernel_dim,
                                     stride=self.convkernel_stride,
                                     padding = self.convInput_padding)
        self.resconv2d_2 = nn.Conv2d(self.conv3_output_dim, self.conv6_output_dim, self.convkernel_dim,
                                     stride=self.convkernel_stride,
                                     padding=self.convInput_padding)
        self.resbatch2d_1 = nn.BatchNorm2d(self.conv3_output_dim)
        self.resbatch2d_2 = nn.BatchNorm2d(self.conv6_output_dim)
        self.batch2d_norm3 = nn.BatchNorm2d(self.conv6_output_dim)
        self.fc1 = nn.Linear(self.fc1_input_dim, self.fc1_output_dim)
        self.fc2 = nn.Linear(self.fc2_input_dim, self.fc2_output_dim)
        self.fc3 = nn.Linear(self.fc3_input_dim, 10)

    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        # x = [128, 3, 32, 32]
        # out = [128, 30, 32, 32]

        out = self.conv2d_1(x)
        out = self.activation(out)
        out = self.batch2d_norm1(out)
        res1 = self.resconv2d_1(out)
        res1 = self.resbatch2d_1(res1)

        out = self.conv2d_2(out)
        out = self.activation(out)
        out = self.conv2d_3(out)
        out = self.activation(out+res1)
        out = self.batch2d_norm2(out)
        res2 = self.resconv2d_2(out)
        res2 = self.resbatch2d_2(res2)

        out = self.conv2d_4(out)
        out = self.activation(out)
        out = self.conv2d_5(out)
        out = self.activation(out)
        out = self.conv2d_6(out)
        out = self.activation(out+res2)
        out = self.batch2d_norm3(out)

        flattened_dim = out.size()[1]*out.size()[2]*out.size()[3]
        flat_out = torch.reshape(out, (len(out),flattened_dim))
        out = self.fc1(flat_out)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.activation(out)
        outs = self.fc3(out)
        # out = self.activation(out)
        # outs = self.fc4(out)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outs