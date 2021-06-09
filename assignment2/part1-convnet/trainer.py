from __future__ import print_function
import numpy as np


class ClassifierTrainer(object):
    """ The trainer class performs SGD with momentum on a cost function """

    def __init__(self):
        self.step_cache = {}  # for storing velocities in momentum update

    def train(self, X, y,
              model, learning_rate_decay=0.95, sample_batches=True,
              num_epochs=30, batch_size=100, acc_frequency=None,
              verbose=False, optimizer=None):
        """
        Optimize the parameters of a model to minimize a loss function. We use
        training data X and y to compute the loss and gradients, and periodically
        check the accuracy on the validation set.

        Inputs:
        - X: Array of training data; each X[i] is a training sample.
        - y: Vector of training labels; y[i] gives the label for X[i].
        - model: Model of ConvNet
        - learning_rate_decay: The learning rate is multiplied by this after each
          epoch.
        - sample_batches: If True, use a minibatch of data for each parameter update
          (stochastic gradient descent); if False, use the entire training set for
          each parameter update (gradient descent).
        - num_epochs: The number of epochs to take over the training data.
        - batch_size: The number of training samples to use at each iteration.
        - acc_frequency: If set to an integer, we compute the training and
          validation set error after every acc_frequency iterations.
        - verbose: If True, print status after each epoch.

        Returns a tuple of:
        - loss_history: List containing the value of the loss function at each
          iteration.
        - train_acc_history: List storing the training set accuracy at each epoch.
        """

        N = X.shape[0]

        if sample_batches:
            iterations_per_epoch = N // batch_size  # using SGD
        else:
            iterations_per_epoch = 1  # using GD
        num_iters = num_epochs * iterations_per_epoch
        epoch = 0
        loss_history = []
        train_acc_history = []
        for it in range(num_iters):
            # get batch of data
            if sample_batches:
                batch_mask = np.random.choice(N, batch_size)
                X_batch = X[batch_mask]
                y_batch = y[batch_mask]
            else:
                # no SGD used, full gradient descent
                X_batch = X
                y_batch = y

            # evaluate cost and gradient
            out, cost = model.forward(X_batch, y_batch)
            model.backward()
            optimizer.update(model)
            loss_history.append(cost)

            # every epoch perform an evaluation on the validation set
            first_it = (it == 0)
            epoch_end = (it + 1) % iterations_per_epoch == 0
            acc_check = (acc_frequency is not None and it % acc_frequency == 0)

            if first_it or epoch_end or acc_check:
                if it > 0 and epoch_end:
                    # decay the learning rate
                    optimizer.learning_rate *= learning_rate_decay
                    epoch += 1

                # evaluate train accuracy
                if N > 1000:
                    train_mask = np.random.choice(N, 1000)
                    X_train_subset = X[train_mask]
                    y_train_subset = y[train_mask]
                else:
                    X_train_subset = X
                    y_train_subset = y

                scores_train, _ = model.forward(X_train_subset, y_train_subset)
                y_pred_train = np.argmax(scores_train, axis=1)
                train_acc = np.mean(y_pred_train == y_train_subset)
                train_acc_history.append(train_acc)

                # print progress if needed
                if verbose:
                    print('Finished epoch %d / %d: cost %f, train: %f, lr %e'
                          % (epoch, num_epochs, cost, train_acc, optimizer.learning_rate))

        # return the training history statistics
        return loss_history, train_acc_history