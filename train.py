import numpy
import theano
import theano.tensor as T
import os
from sklearn.cross_validation import train_test_split
import pickle
import random
import itertools
from theano.tensor.nnet import sigmoid
import scipy.sparse
import h5py

rng = numpy.random

def floatX(x):
    return numpy.asarray(x, dtype=theano.config.floatX)

def load_data(dir='/mnt/games'):
    for fn in os.listdir(dir):
        if not fn.endswith('.hdf5'):
            continue

        fn = os.path.join(dir, fn)
        yield h5py.File(fn)


def get_data():
    X, Xr = [], []
    for f in load_data():
        X.append(f['x'].value)
        Xr.append(f['xr'].value)

    X = numpy.vstack(X)
    Xr = numpy.vstack(Xr)

    test_size = min(0.01, 10000.0 / len(X))
    print 'Splitting', len(X), 'entries into train/test set'
    X_train, X_test, Xr_train, Xr_test = train_test_split(X, Xr, test_size=test_size)

    print X_train.shape[0], 'train set', X_test.shape[0], 'test set'
    return X_train, X_test, Xr_train, Xr_test


def get_parameters(n_in=None, n_hidden_units=2048, n_hidden_layers=None, Ws=None, bs=None):
    if Ws is None or bs is None:
        print 'initializing Ws & bs'
        if type(n_hidden_units) != list:
            n_hidden_units = [n_hidden_units] * n_hidden_layers
        else:
            n_hidden_layers = len(n_hidden_units)

        Ws = []
        bs = []

        def W_values(n_in, n_out):
            return numpy.asarray(rng.uniform(
                low=-numpy.sqrt(6. / (n_in + n_out)),
                high=numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype=theano.config.floatX)

        
        for l in xrange(n_hidden_layers):
            if l == 0:
                n_in_2 = n_in
            else:
                n_in_2 = n_hidden_units[l-1]
            if l < n_hidden_layers - 1:
                n_out_2 = n_hidden_units[l]
                W = W_values(n_in_2, n_out_2)
                b = numpy.zeros(n_out_2, dtype=theano.config.floatX)
            else:
                W = numpy.zeros(n_in_2, dtype=theano.config.floatX)
                b = floatX(0.)
            Ws.append(W)
            bs.append(b)

    Ws_s = [theano.shared(W) for W in Ws]
    bs_s = [theano.shared(b) for b in bs]

    return Ws_s, bs_s


def get_model(Ws_s, bs_s, dropout=False):
    print 'building expression graph'
    x_s = T.matrix('x')

    # Convert input into a 12 * 64 list
    pieces = []
    for piece in [1,2,3,4,5,6, 8,9,10,11,12,13]:
        # pieces.append((x_s <= piece and x_s >= piece).astype(theano.config.floatX))
        pieces.append(T.eq(x_s, piece))

    binary_layer = T.concatenate(pieces, axis=1)

    srng = theano.tensor.shared_randomstreams.RandomStreams(
        rng.randint(999999))

    last_layer = binary_layer
    n = len(Ws_s)
    for l in xrange(n - 1):
        # h = T.tanh(T.dot(last_layer, Ws[l]) + bs[l])
        h = T.dot(last_layer, Ws_s[l]) + bs_s[l]
        h = h * (h > 0)
        
        if dropout:
            mask = srng.binomial(n=1, p=0.5, size=h.shape)
            last_layer = h * T.cast(mask, theano.config.floatX) * 2
        last_layer = h

    p_s = T.dot(last_layer, Ws_s[-1]) + bs_s[-1]
    return x_s, p_s


def get_training_model(Ws_s, bs_s, dropout=False, lambd=1.0):
    # Build a dual network, one for the real move, one for a fake random move
    # Train on a negative log likelihood of classifying the right move

    x_s, x_p = get_model(Ws_s, bs_s, dropout=dropout)
    xr_s, xr_p = get_model(Ws_s, bs_s, dropout=dropout)

    loss = -T.log(sigmoid(x_p - xr_p)).mean() # negative log likelihood

    # Add regularization terms
    reg = 0
    for W in Ws_s:
        reg += lambd * abs(W).mean()
    for b in bs_s:
        reg += lambd * abs(b).mean()

    return x_s, xr_s, loss, reg


def nesterov_updates(loss, all_params, learn_rate, momentum):
    updates = []
    all_grads = T.grad(loss, all_params)
    for param_i, grad_i in zip(all_params, all_grads):
        # generate a momentum parameter
        mparam_i = theano.shared(
            numpy.array(param_i.get_value()*0., dtype=theano.config.floatX))
        v = momentum * mparam_i - learn_rate * grad_i
        w = param_i + momentum * v - learn_rate * grad_i
        updates.append((param_i, w))
        updates.append((mparam_i, v))
    return updates


def get_function(Ws_s, bs_s, dropout=False, update=False, learning_rate=None):
    x_s, xr_s, loss_f, reg_f = get_training_model(Ws_s, bs_s, dropout=dropout)
    obj_f = loss_f + reg_f

    momentum = floatX(0.9)

    if update:
        updates = nesterov_updates(obj_f, Ws_s + bs_s, learning_rate, momentum)
    else:
        updates = []

    print 'compiling function'
    f = theano.function(
        inputs=[x_s, xr_s],
        outputs=[loss_f, reg_f],
        updates=updates,
        on_unused_input='warn')

    return f

def train():
    X_train, X_test, Xr_train, Xr_test = get_data()
    n_in = 12 * 64

    Ws_s, bs_s = get_parameters(n_in=n_in, n_hidden_units=[2048,2048,2048,2048])
    
    minibatch_size = min(1000, X_train.shape[0])

    learning_rate = floatX(0.1)
    while True:
        print 'learning rate:', learning_rate

        train = get_function(Ws_s, bs_s, dropout=False, update=True, learning_rate=learning_rate)
        test = get_function(Ws_s, bs_s, dropout=False, update=False)

        # Train
        best_test_loss = float('inf')
        best_i = None

        for i in itertools.count():
            minibatch_index = random.randint(0, int(X_train.shape[0] / minibatch_size) - 1)
            lo, hi = minibatch_index * minibatch_size, (minibatch_index + 1) * minibatch_size
            loss, reg = train(X_train[lo:hi], Xr_train[lo:hi])
            zs = [loss, reg]
            zs.append(sum(zs))
            test_loss, test_reg = test(X_test, Xr_test)
            print 'iteration %6d %s test %12.9f' % (i, '\t'.join(['%12.9f' % z for z in zs]), test_loss)

            if test_loss < best_test_loss:
                print 'new record!'
                best_test_loss = test_loss
                best_i = i

            if i > best_i + 400 and learning_rate > 0.01:
                print 'no improvements for a long time'
                break
        
            if (i+1) % 100 == 0:
                print 'dumping pickled model'
                f = open('model.pickle', 'w')
                def values(zs):
                    return [z.get_value(borrow=True) for z in zs]
                pickle.dump((values(Ws_s), values(bs_s)), f)
                f.close()

        learning_rate *= floatX(0.5)

if __name__ == '__main__':
    train()
