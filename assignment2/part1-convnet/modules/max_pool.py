import numpy as np

class MaxPooling:
    '''
    Max Pooling of input
    '''
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None
        self.dx = None

    def forward(self, x):
        '''
        Forward pass of max pooling
        :param x: input, (N, C, H, W)
        Number of images in batch
        Number of channels
        Height of Image
        Width of Image
        :return: The output by max pooling with kernel_size and stride
        '''
        out = None
        #############################################################################
        # TODO: Implement the max pooling forward pass.                             #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #############################################################################
        # for kernel size, create container each iteration.
        # then loop thru: ROUNDDOWN( column number / (kernel_size * stride number) )

        # compute iterations horizontal
        numElements = x.shape[3]
        remainder = 1000
        c_range = 0
        while(remainder > 0):
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

        # establish shape.
        maxes = np.empty((x.shape[0], x.shape[1], r_range, c_range))
        container = np.empty((self.kernel_size, self.kernel_size))

        # initialize indices.
        index_x = np.empty((0,0))
        index_y = np.empty((0,0))
        # iterate over number of images
        for n in range(0,x.shape[0] ):
            #iterate over number of channels
            for ch in range(0,x.shape[1]):
                # iterate over rows and cols.
                for r in range (0, r_range):
                    r_real = int(r * self.stride)
                    for c in range(0, c_range):
                        c_real = int(c * self.stride)
                        # for this certain row and col, fill in container and find the max.
                        for i in range (0, self.kernel_size):
                            for j in range (0,self.kernel_size):
                                container[i][j] = x[n][ch][r_real + i][c_real+ j]
                        # store max into max array.
                        maxes[n][ch][r][c] = np.amax(container)
                        ind_container = np.unravel_index(np.argmax(container,axis=None), container.shape)
                        index_x = np.append(index_x, ind_container[0] + r_real)
                        index_y = np.append(index_y, ind_container[1] + c_real)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        self.cache = (x, index_x, index_y)
        out = maxes
        return out

    def backward(self, dout):
        '''
        Backward pass of max pooling
        :param dout: Upstream derivatives
        :return:
        '''
        x, index_x, index_y = self.cache
        #############################################################################
        # TODO: Implement the max pooling backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                     #
        #       2) You may find np.unravel_index useful                             #
        #############################################################################
        dLdy = dout
        dydx = 0
        dLdx = dLdy * dydx
        idx = 0
        # iterate over number of images
        dx = np.zeros((x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
        for n in range(0, x.shape[0]):
            # iterate over number of channels
            for ch in range(0, x.shape[1]):
                # iterate over rows and cols.
                for r in range(0, dout.shape[2]):
                    for c in range(0, dout.shape[3]):
                        idx1 = int(index_x[idx])
                        idx2 = int(index_y[idx])
                        dx[n][ch][idx1][idx2] = dout[n][ch][r][c]
                        idx = idx + 1

        self.dx = dx
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
