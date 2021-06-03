# Do not use packages that are not in standard distribution of python
import numpy as np
np.random.seed(1024)
from ._base_network import _baseNetwork

class TwoLayerNet(_baseNetwork):
    def __init__(self, input_size=28 * 28, num_classes=10, hidden_size=128):
        super().__init__(input_size, num_classes)

        self.hidden_size = hidden_size
        self._weight_init()


    def _weight_init(self):
        '''
        initialize weights of the network
        :return: None; self.weights is filled based on method
        - W1: The weight matrix of the first layer of shape (num_features, hidden_size)
        - b1: The bias term of the first layer of shape (hidden_size,)
        - W2: The weight matrix of the second layer of shape (hidden_size, num_classes)
        - b2: The bias term of the second layer of shape (num_classes,)
        '''

        # initialize weights
        self.weights['b1'] = np.zeros(self.hidden_size)
        self.weights['b2'] = np.zeros(self.num_classes)
        np.random.seed(1024)
        self.weights['W1'] = 0.001 * np.random.randn(self.input_size, self.hidden_size)
        np.random.seed(1024)
        self.weights['W2'] = 0.001 * np.random.randn(self.hidden_size, self.num_classes)

        # initialize gradients to zeros
        self.gradients['W1'] = np.zeros((self.input_size, self.hidden_size))
        self.gradients['b1'] = np.zeros(self.hidden_size)
        self.gradients['W2'] = np.zeros((self.hidden_size, self.num_classes))
        self.gradients['b2'] = np.zeros(self.num_classes)

    def forward(self, X, y, mode='train'):
        '''
        The forward pass of the two-layer net. The activation function used in between the two layers is sigmoid, which
        is to be implemented in self.,sigmoid.
        The method forward should compute the loss of input batch X and gradients of each weights.
        Further, it should also compute the accuracy of given batch. The loss and
        accuracy are returned by the method and gradients are stored in self.gradients

        :param X: a batch of images (N, input_size)
        :param y: labels of images in the batch (N,)
        :param mode: if mode is training, compute and update gradients;else, just return the loss and accuracy
        :return:
            loss: the loss associated with the batch
            accuracy: the accuracy of the batch
            self.gradients: gradients are not explicitly returned but rather updated in the class member self.gradients
        '''
        loss = None
        accuracy = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the forward process:                                      #
        #        1) Call sigmoid function between the two layers for non-linearity  #
        #        2) The output of the second layer should be passed to softmax      #
        #        function before computing the cross entropy loss                   #
        #    2) Compute Cross-Entropy Loss and batch accuracy based on network      #
        #       outputs                                                             #
        #############################################################################
        weightss1 = self.weights['W1']
        weightss2 = self.weights['b1']
        weightss3 = self.weights['W2']
        weightss4 = self.weights['b2']

        z = np.matmul(X ,self.weights['W1']) + self.weights['b1']           # compute z based off first weights and bias. (64*784)*(784*128) + 128*1 . X= 64*784. 64 batch. 784 per batch. transform 784->128. 64*128
        k = self.sigmoid(z)                                                 # compute z based off sigmoid (in between)
        t = np.matmul(k ,self.weights['W2']) + self.weights['b2']           # compute z based off second weights and bias
        prob_t = self.softmax(t)                                            # compute softmax.
        loss = self.cross_entropy_loss(prob_t, y)                           # compute cross entropy loss.
        accuracy = self.compute_accuracy(prob_t, y)                         # compute accuracy
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the backward process:                                     #
        #        1) Compute gradients of each weight and bias by chain rule         #
        #        2) Store the gradients in self.gradients                           #
        #    HINT: You will need to compute gradients backwards, i.e, compute       #
        #          gradients of W2 and b2 first, then compute it for W1 and b1      #
        #          You may also want to implement the analytical derivative of      #
        #          the sigmoid function in self.sigmoid_dev first                   #
        #############################################################################

        # deriving cross entropy derivative
        dLdt = prob_t
        dLdt[range(y.shape[0]), y] -= 1
        dLdt = dLdt / y.shape[0]

        # deriving derivatives with respect to bias 2 and weights 2.
        dtdb2 = np.ones(dLdt.shape[0])
        dtdW2 = k.transpose()

        # deriving chain rule derivatives for derivatives of bias 1 and weights 1.
        dtdk = self.weights['W2'].transpose()
        dkdz = self.sigmoid_dev(z)
        dzdW1 = X.transpose()
        dzdb1 = np.ones(X.shape[0])

        # applying chain rule to solve for derivatives of bias 1 and weights 1.
        dLdk = np.matmul(dLdt,dtdk)
        dLdz = np.multiply(dLdk, dkdz)
        dLdW1 = np.matmul(dzdW1,dLdz)
        dLdb1 = np.matmul(dLdz.transpose(), dzdb1)

        # applying chain rule to solve for derivatives of bias 2 and weights 2.
        dLdW2 = np.matmul(dtdW2,dLdt)
        dLdb2 = np.matmul(dLdt.transpose() , dtdb2)

        #
        self.gradients['W1'] = dLdW1
        self.gradients['b1'] = dLdb1
        self.gradients['W2'] = dLdW2
        self.gradients['b2'] = dLdb2

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return loss, accuracy


