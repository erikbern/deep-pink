import numpy
import theano
import theano.tensor as T
import os
from sklearn.cross_validation import train_test_split
import pickle
import random
import itertools
from theano.tensor.nnet import sigmoid
from train import floatX, load_data, get_data, get_parameters, get_model, nesterov_updates

MINIBATCH_SIZE = 1000


def get_training_model(Ws_s, bs_s, dropout=False, lambd=1.0):
    x_s, x_p = get_model(Ws_s, bs_s, dropout=dropout) # For current position
    xp_s, xp_p = get_model(Ws_s, bs_s, dropout=dropout) # For parent

    y = T.vector('y') # whether current player wins

    t = 0.5 * (y + 1) # change scale to {0, 0.5, 1.0}

    loss_a = -t * T.log(sigmoid(x_p)) - (1-t) * T.log(sigmoid(-x_p))
    loss_b = -t * T.log(sigmoid(-xp_p)) - (1-t) * T.log(sigmoid(xp_p))

    loss = 0.5 * (loss_a + loss_b).mean()

    reg = 0
    for W in Ws_s:
        reg += lambd * abs(W).mean()
    for b in bs_s:
        reg += lambd * abs(b).mean()

    return x_s, xp_s, y, loss, reg

def get_function(Ws_s, bs_s, dropout=False, update=False, learning_rate=None):
    x_s, xp_s, y, loss, reg = get_training_model(Ws_s, bs_s, dropout=dropout)
    obj = loss + reg

    momentum = floatX(0.9)

    if update:
        updates = nesterov_updates(obj, Ws_s + bs_s, learning_rate, momentum)
    else:
        updates = []

    print 'compiling function'
    f = theano.function(
        inputs=[x_s, xp_s, y],
        outputs=[loss, reg],
        updates=updates,
        on_unused_input='warn')

    return f

def train():
    X_train, X_test, Xp_train, Xp_test, Y_train, Y_test = get_data(['x', 'xp', 'y'])
    n_in = 12 * 64

    f = open('model.pickle')
    Ws, bs = pickle.load(f)

    # Initialize the last layer to zero to re-scale it
    Ws[-1] *= 0.0
    bs[-1] *= 0.0

    Ws_s, bs_s = get_parameters(Ws=Ws, bs=bs)

    minibatch_size = min(MINIBATCH_SIZE, X_train.shape[0])

    for n_iterations, learning_rate in [(20000, 1e-3), (20000, 1e-4), (20000, 1e-5)]:
        learning_rate = floatX(learning_rate)
        print 'learning rate:', learning_rate

        train = get_function(Ws_s, bs_s, dropout=False, update=True, learning_rate=learning_rate)
        test = get_function(Ws_s, bs_s, dropout=False, update=False)

        # Train
        best_test_loss = float('inf')

        for i in xrange(n_iterations):
            minibatch_index = random.randint(0, int(X_train.shape[0] / minibatch_size) - 1)
            lo, hi = minibatch_index * minibatch_size, (minibatch_index + 1) * minibatch_size
            loss, reg = train(X_train[lo:hi], Xp_train[lo:hi], Y_train[lo:hi])
            zs = [loss, reg]
            zs.append(sum(zs))
            test_loss, test_reg = test(X_test, Xp_test, Y_test)
            print 'iteration %6d %s test %12.9f' % (i, '\t'.join(['%12.9f' % z for z in zs]), test_loss)

            if test_loss < best_test_loss:
                print 'new record!'
                best_test_loss = test_loss

            if (i+1) % 100 == 0:
                print 'dumping pickled model'
                f = open('model_y.pickle', 'w')
                def values(zs):
                    return [z.get_value(borrow=True) for z in zs]
                pickle.dump((values(Ws_s), values(bs_s)), f)
                f.close()


if __name__ == '__main__':
    train()
