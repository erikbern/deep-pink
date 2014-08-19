import numpy
import theano
import theano.tensor as T
import os
from sklearn.cross_validation import train_test_split
import pickle
import random
import itertools
from theano.tensor.nnet import sigmoid

rng = numpy.random

def floatX(x):
    return numpy.asarray(x, dtype=theano.config.floatX)

def load_data(dir='/mnt/games'):
    X, Xp, Xr, M, Y = [], [], [], [], []

    def read_stuff():
        for fn in os.listdir(dir):
            if not fn.endswith('_2.npy'):
                continue

            fn = os.path.join(dir, fn)
            print fn
            f = open(fn)

            while True:
                try:
                    x, xp, xr, m, y = [numpy.load(f) for i in xrange(5)]
                except:
                    break
                
                yield x, xp, xr, m, y

    for x, xp, xr, m, y in read_stuff():
        X.append(x)
        Xr.append(xr)

        if len(X) % 1000 == 0:
            print len(X), '...'

        if len(X) == 1000000:
            break

    return X, Xr


def get_data():
    X, Xr = [numpy.array(x) for x in load_data()]

    test_size = min(0.01, 10000.0 / len(X))
    X_train, X_test, Xr_train, Xr_test = train_test_split(X, Xr, test_size=test_size)

    print len(X_train), 'train set', len(X_test), 'test set'

    return X_train, X_test, Xr_train, Xr_test


def get_parameters(n_in=None, n_hidden_units=2048, n_hidden_layers=4, Ws=None, bs=None):
    if Ws is None or bs is None:
        print 'initializing Ws & bs'
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
                n_in_2 = n_hidden_units
            if l < n_hidden_layers - 1:
                n_out_2 = n_hidden_units
                W = W_values(n_in_2, n_out_2)
                b = numpy.ones(n_out_2, dtype=theano.config.floatX)
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

    srng = theano.tensor.shared_randomstreams.RandomStreams(
        rng.randint(999999))

    last_layer = x_s
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


def get_training_model(Ws_s, bs_s, dropout=False, lambd=0.1):
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


def get_function(Ws_s, bs_s, dropout=False, update=False, learning_rate=None):
    x_s, xr_s, loss_f, reg_f = get_training_model(Ws_s, bs_s, dropout=dropout)
    obj_f = loss_f + reg_f

    momentum = floatX(0.9)

    updates = []
    if update:
        for param in Ws_s + bs_s:
            param_update = theano.shared(numpy.zeros(param.get_value().shape, dtype=theano.config.floatX))
            updates.append((param, param - learning_rate * param_update))
            updates.append((param_update, momentum * param_update + floatX(1. - momentum) * T.grad(obj_f, param)))

    print 'compiling function'
    f = theano.function(
        inputs=[x_s, xr_s],
        outputs=[loss_f, reg_f],
        updates=updates,
        on_unused_input='warn')

    return f

def train():
    X_train, X_test, Xr_train, Xr_test = get_data()
    n_in = len(X_train[0])

    Ws_s, bs_s = get_parameters(n_in)
    
    minibatch_size = min(10000, len(X_train))

    learning_rate = floatX(1.0)
    while True:
        print 'learning rate:', learning_rate

        train = get_function(Ws_s, bs_s, dropout=False, update=True, learning_rate=learning_rate)
        test = get_function(Ws_s, bs_s, dropout=False, update=False)

        # Train
        best_test_loss = float('inf')
        best_i = None

        for i in itertools.count():
            minibatch_index = random.randint(0, int(len(X_train) / minibatch_size) - 1)
            lo, hi = minibatch_index * minibatch_size, (minibatch_index + 1) * minibatch_size
            loss, reg = train(X_train[lo:hi], Xr_train[lo:hi])
            zs = [loss, reg]
            zs.append(sum(zs))
            test_loss, test_reg = test(X_test, Xr_test)
            print 'iteration %6d %s test %12.9f' % (i, '\t'.join(['%12.9f' % z for z in zs]), test_loss)

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_i = i

            if i > best_i + 1000 and learning_rate > 1e-3:
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
