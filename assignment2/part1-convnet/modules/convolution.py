import numpy as np

class Conv2D:
    '''
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        '''
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        '''
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.out_channels, self.in_channels,  self.kernel_size, self.kernel_size)
        self.bias = np.zeros(self.out_channels)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        '''
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters. Save necessary variables in self.cache for backward pass
        '''
        out = None
        #############################################################################
        # TODO: Implement the convolution forward pass.                             #
        # Hint: 1) You may use np.pad for padding.                                  #
        #       2) You may implement the convolution with loops                     #
        #############################################################################
        self.cache = x

        x = np.pad(x, ((0,0),(0,0),(self.padding, self.padding),(self.padding, self.padding)), 'constant')
        # compute iterations horizontal
        numElements = x.shape[3]
        remainder = 1000
        c_range = 0
        while (remainder > 0):
            remainder = numElements - self.kernel_size
            numElements = numElements - self.stride
            c_range = c_range + 1
        if x.shape[3] - self.kernel_size < 0:
            c_range = 0

        # compute iterations vertical
        remainder = 1000
        numElements = x.shape[2]
        r_range = 0
        while (remainder > 0):
            remainder = numElements - self.kernel_size
            numElements = numElements - self.stride
            r_range = r_range + 1
        if x.shape[2] - self.kernel_size < 0:
            r_range = 0

        # output size.
        out = np.zeros((x.shape[0], self.out_channels, r_range, c_range))

        # iterate over number of images
        for n in range(0, x.shape[0]):
            # iterate over number of channels
            for ch in range(0, x.shape[1]):
                # iterate over rows and cols, stride as necessary.
                for r in range(0, r_range):
                    r_real = int(r * self.stride)
                    for c in range(0, c_range):
                        c_real = int(c * self.stride)
                        # for this certain row and col, perform convolution across all channels.
                        for filt_idx in range(0,self.out_channels):
                            for i in range(0, self.kernel_size):
                                for j in range(0, self.kernel_size):
                                    out[n][filt_idx][r][c] =  out[n][filt_idx][r][c] + x[n][ch][r_real + i][c_real + j]*self.weight[filt_idx][ch][i][j]

        # add bias term where necessary.
        for n in range(0, x.shape[0]):
            for filt_idx in range(0, self.out_channels):
                for r in range(0, r_range):
                    for c in range(0,c_range):
                        out[n][filt_idx][r][c] = out[n][filt_idx][r][c] + self.bias[filt_idx]
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out

    def backward(self, dout):
        '''
        The backward pass of convolution
        :param dout: upstream gradients
        :return: nothing but dx, dw, and db of self should be updated
        '''
        x = self.cache
        x_pad = np.pad(x, ((0,0),(0,0),(self.padding, self.padding),(self.padding, self.padding)), 'constant')

        #############################################################################
        # TODO: Implement the convolution backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the convolution with loops                     #
        #       2) don't forget padding when computing dx                           #
        #############################################################################

        #compute dLdb.
        db = np.zeros((self.bias.shape))
        # iterate over number of channels
        for ch in range(0, dout.shape[1]):
            # iterate over number of images
            for n in range(0, dout.shape[0]):
                # iterate over rows and cols
                for i in range(0, dout.shape[2]):
                    for j in range(0, dout.shape[3]):
                        db[ch] = db[ch] + dout[n][ch][i][j]


        #compute dLdk

        # compute iterations horizontal [for striding]
        numElements = x_pad.shape[3]
        remainder = 1000
        c_range = 0
        while (remainder > 0):
            remainder = numElements - self.kernel_size
            numElements = numElements - self.stride
            c_range = c_range + 1
        if x_pad.shape[3] - self.kernel_size < 0:
            c_range = 0

        # compute iterations vertical [for striding]
        remainder = 1000
        numElements = x_pad.shape[2]
        r_range = 0
        while (remainder > 0):
            remainder = numElements - self.kernel_size
            numElements = numElements - self.stride
            r_range = r_range + 1
        if x_pad.shape[2] - self.kernel_size < 0:
            r_range = 0

        dk = np.zeros((self.weight.shape))
        # for each filter
        for filt_idx in range(0, self.out_channels):
            # for each image
            for n in range(0, x.shape[0]):
                # for each channel within the image
                for ch in range(0, self.in_channels):
                    # for a selected (a', b') weight.
                    for a in range(0, self.weight.shape[2]):
                        for b in range(0, self.weight.shape[3]):
                            # compute gradient by summing over x's. stride if necessary.
                            for r in range(0, r_range):
                                r_real = int(r * self.stride)
                                for c in range(0, c_range):
                                    c_real = int(c * self.stride)
                                    dk[filt_idx][ch][a][b] = dk[filt_idx][ch][a][b] + dout[n][filt_idx][r][c] * x_pad[n][ch][r_real+a][c_real+b]



        #compute dLdx  dw = (2,3,3,3) ... w = [Cout, Cin, Hk, Wk]
        #              x  = (4,3,5,5)     X = [N, Cin, Hi, Wi]
        #            dout = (4,2,5,5)     Y = [N, Cout,Hi, Wi]
        dx_pad = np.zeros((x_pad.shape))

        #for each image
        for n in range(0, x.shape[0]):
            #for each channel
            for ch in range(0, self.in_channels):
                #for each row and column within the image channel.
                for r in range(0, x_pad.shape[2]):
                    r_real = int(r)
                    for c in range(0, x_pad.shape[3]):
                        c_real = int(c)
                        #for each weight in image.
                        for a in range(0,self.weight.shape[2]):
                            for b in range(0, self.weight.shape[3]):
                                for filt_idx in range(0, self.out_channels):
                                    if (r_real - a*self.stride >= 0) and (c_real - b*self.stride >= 0) and (r_real - a*self.stride < dout.shape[2]) and (c_real - b*self.stride < dout.shape[3]):
                                        dx_pad[n][ch][r][c] = dx_pad[n][ch][r][c] + dout[n][filt_idx][r - a*self.stride][c - b*self.stride] * self.weight[filt_idx][ch][a][b]


        slices = []
        pad_width = ((0,0),(0,0),(self.padding, self.padding),(self.padding, self.padding))
        for c in pad_width:
            e = None if c[1] == 0 else -c[1]
            slices.append(slice(c[0], e))
        dx = dx_pad[tuple(slices)]
        self.dw = dk
        self.dx = dx
        self.db = db


        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################