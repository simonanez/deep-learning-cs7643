import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self,resconv2d_1, resbatch2d_1, resconv2d_2, resbatch2d_2, resconv2d_3, resbatch2d_3):
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
        self.conv1_output_dim = 32

        self.conv2_input_dim = 32
        self.conv2_output_dim = 32

        self.conv21_input_dim = 32
        self.conv21_output_dim = 32

        self.conv3_input_dim = 32
        self.conv3_output_dim = 64

        self.conv31_input_dim = 64
        self.conv31_output_dim = 64

        self.conv4_input_dim = 64
        self.conv4_output_dim = 64

        self.conv5_input_dim = 64
        self.conv5_output_dim = 128

        self.conv51_input_dim = 128
        self.conv51_output_dim = 128

        self.conv6_input_dim = 128
        self.conv6_output_dim = 128

        self.fc1_input_dim = 2048
        self.fc1_output_dim = 128
        self.fc2_input_dim = 128
        self.fc2_output_dim = 10

        self.batch2d_norm0 = nn.BatchNorm2d(3)

        self.conv2d_1 = nn.Conv2d(self.conv1_input_dim, self.conv1_output_dim, self.convkernel_dim,
                            stride=self.convkernel_stride,
                            padding=self.convInput_padding)
        self.batch2d_norm1 = nn.BatchNorm2d(self.conv1_output_dim)
        self.conv2d_2 = nn.Conv2d(self.conv2_input_dim, self.conv2_output_dim, self.convkernel_dim,
                                  stride=self.convkernel_stride,
                                  padding=self.convInput_padding)
        self.batch2d_norm2 = nn.BatchNorm2d(self.conv2_output_dim)
        self.conv2d_21 = nn.Conv2d(self.conv21_input_dim, self.conv21_output_dim, self.convkernel_dim,
                                  stride=self.convkernel_stride,
                                  padding=self.convInput_padding)
        self.batch2d_norm21 = nn.BatchNorm2d(self.conv21_output_dim)



        self.conv2d_3 = nn.Conv2d(self.conv3_input_dim, self.conv3_output_dim, self.convkernel_dim,
                                  stride=self.convkernel_stride,
                                  padding=self.convInput_padding)
        self.batch2d_norm3 = nn.BatchNorm2d(self.conv3_output_dim)
        self.conv2d_31 = nn.Conv2d(self.conv31_input_dim, self.conv31_output_dim, self.convkernel_dim,
                                   stride=self.convkernel_stride,
                                   padding=self.convInput_padding)
        self.batch2d_norm31 = nn.BatchNorm2d(self.conv31_output_dim)
        self.conv2d_4 = nn.Conv2d(self.conv4_input_dim, self.conv4_output_dim, self.convkernel_dim,
                                  stride=self.convkernel_stride,
                                  padding=self.convInput_padding)
        self.batch2d_norm4 = nn.BatchNorm2d(self.conv4_output_dim)



        self.conv2d_5 = nn.Conv2d(self.conv5_input_dim, self.conv5_output_dim, self.convkernel_dim,
                                  stride=self.convkernel_stride,
                                  padding=self.convInput_padding)
        self.batch2d_norm5 = nn.BatchNorm2d(self.conv5_output_dim)
        self.conv2d_51 = nn.Conv2d(self.conv51_input_dim, self.conv51_output_dim, self.convkernel_dim,
                                  stride=self.convkernel_stride,
                                  padding=self.convInput_padding)
        self.batch2d_norm51 = nn.BatchNorm2d(self.conv51_output_dim)

        self.conv2d_6 = nn.Conv2d(self.conv6_input_dim, self.conv6_output_dim, self.convkernel_dim,
                                  stride=self.convkernel_stride,
                                  padding=self.convInput_padding)
        self.batch2d_norm6 = nn.BatchNorm2d(self.conv6_output_dim)
        self.fc1 = nn.Linear(self.fc1_input_dim, self.fc1_output_dim)
        self.fc2 = nn.Linear(self.fc2_input_dim, self.fc2_output_dim)
        # self.fc3 = nn.Linear(self.fc3_input_dim, self.fc3_output_dim)

        self.maxpool1 = nn.MaxPool2d(2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.maxpool3 = nn.MaxPool2d(2)

        #residual networks:
        self.resconv1 = nn.Conv2d(self.conv3_input_dim, self.conv4_output_dim,
                                    self.convkernel_dim,
                                    stride=self.convkernel_stride,
                                    padding=self.convInput_padding)
        self.resconv2 = nn.Conv2d(self.conv4_output_dim, self.conv6_output_dim,
                                  self.convkernel_dim,
                                  stride=self.convkernel_stride,
                                  padding=self.convInput_padding)
        self.resbatch1 = nn.BatchNorm2d(self.conv4_output_dim)
        self.resbatch2 = nn.BatchNorm2d(self.conv6_output_dim)

        # self.fc4 = nn.Linear(self.fc4_input_dim, self.fc4_output_dim)

        self.activation1 = nn.ReLU()
        self.activation2 = nn.ReLU()
        self.activation21 = nn.ReLU()
        self.activation3 = nn.ReLU()
        self.activation31 = nn.ReLU()
        self.activation4 = nn.ReLU()
        self.activation5 = nn.ReLU()
        self.activation51 = nn.ReLU()
        self.activation6 = nn.ReLU()
        self.activation7 = nn.ReLU()
        self.activation8 = nn.ReLU()
        self.activation9 = nn.ReLU()



    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        # x = [128, 3, 32, 32]
        # out = [128, 30, 32, 32]
        # two convolutions
        out = self.conv2d_1(x)
        out = self.activation1(out)
        out = self.batch2d_norm1(out)
        out = self.conv2d_2(out)
        out = self.activation2(out)
        out = self.batch2d_norm2(out)
        out = self.conv2d_21(out)
        out = self.activation21(out)
        out = self.batch2d_norm21(out)
        out = self.maxpool1(out)
        res1 = self.resconv1(out)
        res1 = self.resbatch1(res1)

        # two convolutions
        out = self.conv2d_3(out)
        out = self.activation3(out)
        out = self.batch2d_norm3(out)
        out = self.conv2d_31(out)
        out = self.activation31(out)
        out = self.batch2d_norm31(out)
        out = self.conv2d_4(out)
        out = self.activation4(out + res1)
        out = self.batch2d_norm4(out)
        out= self.maxpool2(out)
        res2 = self.resconv2(out)
        res2 = self.resbatch2(res2)

        #two convolutions
        out = self.conv2d_5(out)
        out = self.activation5(out)
        out = self.batch2d_norm5(out)
        out = self.conv2d_51(out)
        out = self.activation51(out)
        out = self.batch2d_norm51(out)
        out = self.conv2d_6(out)
        out = self.activation6(out + res2)
        out = self.batch2d_norm6(out)
        out= self.maxpool3(out)

        # two fully connected networks.
        flattened_dim = out.size()[1]*out.size()[2]*out.size()[3]
        flat_out = torch.reshape(out, (len(out),flattened_dim))
        out = self.fc1(flat_out)
        out = self.activation7(out)
        outs = self.fc2(out)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outs