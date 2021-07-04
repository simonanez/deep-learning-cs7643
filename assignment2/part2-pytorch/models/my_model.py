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

        self.conv3_input_dim = 32
        self.conv3_output_dim = 64

        self.conv4_input_dim = 64
        self.conv4_output_dim = 64

        self.conv5_input_dim = 64
        self.conv5_output_dim = 128

        self.conv6_input_dim = 128
        self.conv6_output_dim = 128

        self.fc1_input_dim = 2048
        self.fc1_output_dim = 128
        self.fc2_input_dim = 128
        self.fc2_output_dim = 10
        # self.fc3_input_dim = 3072
        # self.fc3_output_dim = 10
        # self.fc4_input_dim = 1500
        # self.fc4_output_dim = 10
        self.batch2d_norm0 = nn.BatchNorm2d(3)

        self.conv2d_1 = nn.Conv2d(self.conv1_input_dim, self.conv1_output_dim, self.convkernel_dim,
                            stride=self.convkernel_stride,
                            padding=self.convInput_padding)
        self.batch2d_norm1 = nn.BatchNorm2d(self.conv1_output_dim)
        self.conv2d_2 = nn.Conv2d(self.conv2_input_dim, self.conv2_output_dim, self.convkernel_dim,
                                  stride=self.convkernel_stride,
                                  padding=self.convInput_padding)
        self.batch2d_norm2 = nn.BatchNorm2d(self.conv2_output_dim)

        self.conv2d_3 = nn.Conv2d(self.conv3_input_dim, self.conv3_output_dim, self.convkernel_dim,
                                  stride=self.convkernel_stride,
                                  padding=self.convInput_padding)
        self.batch2d_norm3 = nn.BatchNorm2d(self.conv3_output_dim)

        self.conv2d_4 = nn.Conv2d(self.conv4_input_dim, self.conv4_output_dim, self.convkernel_dim,
                                  stride=self.convkernel_stride,
                                  padding=self.convInput_padding)
        self.batch2d_norm4 = nn.BatchNorm2d(self.conv4_output_dim)
        self.conv2d_5 = nn.Conv2d(self.conv5_input_dim, self.conv5_output_dim, self.convkernel_dim,
                                  stride=self.convkernel_stride,
                                  padding=self.convInput_padding)
        self.batch2d_norm5 = nn.BatchNorm2d(self.conv5_output_dim)
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

        # self.fc4 = nn.Linear(self.fc4_input_dim, self.fc4_output_dim)

        self.activation1 = nn.ReLU()
        self.activation2 = nn.ReLU()
        self.activation3 = nn.ReLU()
        self.activation4 = nn.ReLU()
        self.activation5 = nn.ReLU()
        self.activation6 = nn.ReLU()
        self.activation7 = nn.ReLU()
        self.activation8 = nn.ReLU()
        self.activation9 = nn.ReLU()


        # self.resconv2d_1 = resconv2d_1
        # self.resconv2d_2 = resconv2d_2
        # self.resconv2d_3 = resconv2d_3
        #
        # self.resbatch2d_1 = resbatch2d_1
        # self.resbatch2d_2 = resbatch2d_2
        # self.resbatch2d_3 = resbatch2d_3
        #
        #
        # self.conv2d_7 = nn.Conv2d(self.conv7_input_dim, self.conv7_output_dim, self.convkernel_dim,
        #                           stride=self.convkernel_stride,
        #                           padding=self.convInput_padding)
        # self.conv2d_8 = nn.Conv2d(self.conv8_input_dim, self.conv8_output_dim, self.convkernel_dim,
        #                           stride=self.convkernel_stride,
        #                           padding=self.convInput_padding)



    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        # x = [128, 3, 32, 32]
        # out = [128, 30, 32, 32]
        out = self.conv2d_1(x)
        out = self.activation1(out)
        out = self.batch2d_norm1(out)
        out = self.conv2d_2(out)
        out = self.activation2(out)
        out = self.batch2d_norm2(out)
        out = self.maxpool1(out)

        out = self.conv2d_3(out)
        out = self.activation3(out)
        out = self.batch2d_norm3(out)
        out = self.conv2d_4(out)
        out = self.activation4(out)
        out = self.batch2d_norm4(out)
        out= self.maxpool2(out)

        out = self.conv2d_5(out)
        out = self.activation5(out)
        out = self.batch2d_norm5(out)
        out = self.conv2d_6(out)
        out = self.activation6(out)
        out = self.batch2d_norm6(out)
        out= self.maxpool3(out)


        flattened_dim = out.size()[1]*out.size()[2]*out.size()[3]
        flat_out = torch.reshape(out, (len(out),flattened_dim))
        out = self.fc1(flat_out)
        out = self.activation7(out)
        outs = self.fc2(out)

        # out = self.activation8(out)
        # outs = self.fc3(out)
        # out = self.activation9(out)
        # outs = self.fc4(out)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outs