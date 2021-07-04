import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def reweight(cls_num_list, beta=0.9999):
    '''
    note for steven: class num list = [5000, 2997, 1796, 1077, 645, 387, 232, 139, 83, 50]
    CE Loss:
    0.92, 0.903, 0.596, 0.534, 0.245, 0.053, 0.179, 0.067, 0.000, 0.000

    Implement reweighting by effective numbers
    :param cls_num_list: a list containing # of samples of each class
    :param beta: hyper-parameter for reweighting, see paper for more details
    :return:
    '''
    # per_cls_weights = None
    per_cls_weights = torch.zeros((len(cls_num_list)))
    #############################################################################
    # TODO: reweight each class by effective numbers                            #
    #############################################################################
    # effective numbers.
    for i in range(0, len(cls_num_list)):
        per_cls_weights[i] = (1 - beta ) / (1 - (beta ** cls_num_list[i]) )
    # renormalize such that sum of weights  = num_classes
    sum_weights = torch.sum(per_cls_weights)
    num_classes = len(cls_num_list)
    norm = num_classes / sum_weights
    # apply norm factor to normalize.
    per_cls_weights = norm * per_cls_weights

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return per_cls_weights


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
    def forward(self, input, target):
        '''
        Implement forward of focal loss
        :param input: input predictions (model output)
        :param target: labels
        :return: tensor of focal loss in scalar
        '''
        loss = None
        #############################################################################
        # TODO: Implement forward pass of the focal loss                            #
        #############################################################################

        losses = torch.zeros(input.shape[0])
        accum = torch.zeros(input.shape)
        z_augmented = torch.zeros(input.shape)
        # first, compute z
        for i in range(0, input.shape[0]):
            for j in range(0,input.shape[1]):
                if j == target[i]:
                    z_augmented[i][j] = input[i][j]
                else:
                    z_augmented[i][j] = -input[i][j]

        # calculate sigmoid
        prob = self.sigmoid(z_augmented)

        # fill in losses.
        for i in range(0, input.shape[0]):
            # perform summation.
            accum[i] = torch.log(prob[i]) * (1 - prob[i]) ** self.gamma
            for j in range(0, accum.shape[1]):
                losses[i] = losses[i] + accum[i][j]
            # multiply by weight specific to samples.
            losses[i] = -self.weight[target[i]] * losses[i]

        sum_loss = 0
        for i in range(0, input.shape[0]):
            sum_loss = sum_loss + losses[i]

        loss = sum_loss / input.shape[0]


        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return loss