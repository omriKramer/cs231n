import numpy as np


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]
    num_classes = W.shape[1]
    for i in range(num_train):
        scores = X[i].dot(W)
        scores_exp = np.exp(scores)
        scores_norm = scores_exp / scores_exp.sum()
        loss += -np.log(scores_norm[y[i]])
        for j in range(num_classes):
            delta_i_j = j == y[i]
            dW[:, j] += -X[i] * (delta_i_j - scores_norm[j])

    loss /= num_train
    dW /= num_train

    loss += reg * np.sum(W * W)
    dW += 2 * reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]

    scores = X.dot(W)

    scores -= scores.max(axis=1, keepdims=True)
    scores_exp = np.exp(scores)
    scores_norm = scores_exp / scores_exp.sum(axis=1, keepdims=True)
    loss = -np.log(scores_norm[range(num_train), y]).sum()

    dscores = scores_norm
    dscores[range(num_train), y] -= 1
    dscores /= num_train
    dW = X.T.dot(dscores)

    loss /= num_train
    loss += reg * np.sum(W * W)

    dW += 2 * reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
