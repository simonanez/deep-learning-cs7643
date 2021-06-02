# Do not use packages that are not in standard distribution of python
import numpy as np

from ._base_network import _baseNetwork

class SoftmaxRegression(_baseNetwork):
    def __init__(self, input_size=28*28, num_classes=10):
        '''
        A single layer softmax regression. The network is composed by:
        a linear layer without bias => (optional ReLU activation) => Softmax
        :param input_size: the input dimension
        :param num_classes: the number of classes in total
        '''
        super().__init__(input_size, num_classes)
        self._weight_init()

    def _weight_init(self):
        '''
        initialize weights of the single layer regression network. No bias term included.
        :return: None; self.weights is filled based on method
        - W1: The weight matrix of the linear layer of shape (num_features, hidden_size)
        '''
        np.random.seed(1024)
        self.weights['W1'] = 0.001 * np.random.randn(self.input_size, self.num_classes)
        self.gradients['W1'] = np.zeros((self.input_size, self.num_classes))

    def forward(self, X, y, mode='train'):
        '''
        Compute loss and gradients using softmax with vectorization.

        :param X: a batch of image (N, 28x28)
        :param y: labels of images in the batch (N,)
        :return:
            loss: the loss associated with the batch
            accuracy: the accuracy of the batch
        '''
        loss = None
        gradient = None
        accuracy = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the forward process and compute the Cross-Entropy loss    #
        #    2) Compute the gradient with respect to the loss                       #
        # Hint:                                                                     #
        #   Store your intermediate outputs before ReLU for backwards               #
        #############################################################################
        weightss = self.weights['W1']
        z = np.matmul(X ,self.weights['W1'])            # compute z based off weights
        t = self.ReLU(z)                                # apply ReLU when necessary
        prob_t = self.softmax(t)                        # softmax over all classes
        loss = self.cross_entropy_loss(prob_t,y)        # compute cross entropy loss.
        accuracy = self.compute_accuracy(prob_t, y)     # compute accuracy.

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        if mode != 'train':
            return loss, accuracy

        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the backward process:                                     #
        #        1) Compute gradients of each weight by chain rule                  #
        #        2) Store the gradients in self.gradients                           #
        #############################################################################
        dLdt = prob_t
        dLdt[range(y.shape[0]), y] -= 1 #good?
        dLdt = dLdt/ y.shape[0]
        dtdz = self.ReLU_dev(z)  #good.  (64, 10)
        dLdz = np.multiply(dLdt,dtdz)
        dzdx = X.transpose()     # good  (784, 64)

        dLdw = np.matmul(dzdx,dLdz)
        self.gradients['W1'] = dLdw
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return loss, accuracy





        


