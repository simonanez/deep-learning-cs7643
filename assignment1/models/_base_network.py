# Do not use packages that are not in standard distribution of python
import numpy as np
class _baseNetwork:
    def __init__(self, input_size=28 * 28, num_classes=10):

        self.input_size = input_size
        self.num_classes = num_classes

        self.weights = dict()
        self.gradients = dict()

    def _weight_init(self):
        pass

    def forward(self):
        pass

    def softmax(self, scores):
        '''
        Compute softmax scores given the raw output from the model

        :param scores: raw scores from the model (N, num_classes)
        :return:
            prob: softmax probabilities (N, num_classes)
        '''
        prob = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Calculate softmax scores of input images                            #
        #############################################################################
        maxes = np.max(scores, axis =1)                             # record row maxes
        shifted_scores = (scores.transpose() - maxes).transpose()   # subtract max from each row
        num = np.exp(shifted_scores)                                # e^(score - max(score)) for each row.
        sum_rows = np.sum(num, axis=1)                              # sum e^(xi) across all i in a row.
        prob = num / sum_rows[:,None]                               # probability result.
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return prob

    def cross_entropy_loss(self, x_pred, y):
        '''
        Compute Cross-Entropy Loss based on prediction of the network and labels
        :param x_pred: Probabilities from the model (N, num_classes)
        :param y: Labels of instances in the batch
        :return: The computed Cross-Entropy Loss
        '''
        loss = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement Cross-Entropy Loss                                        #
        #############################################################################
        selected_probabilities = x_pred[ range(x_pred.shape[0]), y]     # select probabilities corresponding to ground truth label.
        log_likelihood = -np.log(selected_probabilities)                # calc losses - log probabilities of all the selected probabilities.
        sum_losses = np.sum(log_likelihood)                             # sum over all log probabilities (losses)
        loss = sum_losses / x_pred.shape[1]                             # expected loss, average loss over all losses.
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return loss

    def compute_accuracy(self, x_pred, y):
        '''
        Compute the accuracy of current batch
        :param x_pred: Probabilities from the model (N, num_classes)
        :param y: Labels of instances in the batch
        :return: The accuracy of the batch
        '''
        acc = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the accuracy function                                     #
        #############################################################################
        y_hat = np.argmax(x_pred, axis=1)           # get predicted value of y.
        count_correct = np.sum(y_hat == y)          # count how many times y predicted matches y actual.
        acc = count_correct / y.shape[0]            # num correct / total number of cases.
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return acc

    def sigmoid(self, X):
        '''
        Compute the sigmoid activation for the input

        :param X: the input data coming out from one layer of the model (N, layer size)
        :return:
            out: the value after the sigmoid activation is applied to the input (N, layer size)
        '''
        #############################################################################
        # TODO: Comput the sigmoid activation on the input                          #
        #############################################################################
        denom = 1 + np.exp(-X)
        out = np.reciprocal(denom)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out

    def sigmoid_dev(self, x):
        '''
        The analytical derivative of sigmoid function at x
        :param x: Input data
        :return: The derivative of sigmoid function at x
        '''
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the derivative of Sigmoid function                        #
        #############################################################################
        ds = self.sigmoid(x) * (1 - self.sigmoid(x))
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return ds

    def ReLU(self, X):
        '''
        Compute the ReLU activation for the input

        :param X: the input data coming out from one layer of the model (N, layer size)
        :return:
            out: the value after the ReLU activation is applied to the input (N, layer size)
        '''
        #############################################################################
        # TODO: Comput the ReLU activation on the input                          #
        #############################################################################
        out = X.clip(min=0)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out

    def ReLU_dev(self,X):
        '''
        Compute the gradient ReLU activation for the input

        :param X: the input data coming out from one layer of the model (N, layer size)
        :return:
            out: gradient of ReLU given input X
        '''
        #############################################################################
        # TODO: Comput the gradient of ReLU activation                              #
        #############################################################################
        out = X.clip(min=0)
        out[out>0] = 1
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out
