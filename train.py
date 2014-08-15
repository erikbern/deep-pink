import numpy
import theano
import theano.tensor as T
import os
from sklearn.cross_validation import train_test_split
import pickle
import random
import itertools

rng = numpy.random

def load_data(dir='games'):
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

        if len(X) == 10000:
            break

    return X, Xr


def get_data():
    X, Xr = [numpy.array(x) for x in load_data()]

    X_train, X_test, Xr_train, Xr_test = train_test_split(X, Xr, test_size=0.01)

    print len(X_train), 'train set', len(X_test), 'test set'

    return X_train, X_test, Xr_train, Xr_test

def get_model(n_in, Ws=[], bs=[], dropout=False, lambd=0.0):
    n_hidden_layers = 2
    n_hidden = 256

    # Declare Theano symbolic variables
    x = T.matrix("x")
    xr = T.matrix("xr")

    def W_values(n_in, n_out):
        return numpy.asarray(rng.uniform(
                low=-numpy.sqrt(6. / (n_in + n_out)),
                high=numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_hidden)), dtype=theano.config.floatX)

    Ws = []
    bs = []

    for l in xrange(n_hidden_layers):
        if l == 0:
            n_in_2 = n_in
        else:
            n_in_2 = n_hidden
        if l < n_hidden_layers - 1:
            n_out_2 = n_hidden
            W = theano.shared(W_values(n_in_2, n_out_2), name="w%d" % l)
            b = theano.shared(numpy.ones(n_out_2) * 1e-2, name="b%d" % l)
        else:
            W = theano.shared(numpy.zeros(n_in_2), name="w%d" % l)
            b = theano.shared(0., name="b%d" % l)

        Ws.append(W)
        bs.append(b)

    # Construct Theano expression graph
    srng = theano.tensor.shared_randomstreams.RandomStreams(
        rng.randint(999999))

    def get_pred(input):
        last_layer = input
        n = len(Ws)
        for l in xrange(n - 1):
            # h = T.tanh(T.dot(last_layer, Ws[l]) + bs[l])
            h = T.dot(last_layer, Ws[l]) + bs[l]
            h = h * (h > 0)

            if dropout:
                mask = srng.binomial(n=1, p=0.5, size=h.shape)
                last_layer = h * T.cast(mask, theano.config.floatX)
            else:
                last_layer = h * 0.5

        # p = T.tanh(T.dot(last_layer, Ws[-1]) + bs[-1])
        p = T.dot(last_layer, Ws[-1]) + bs[-1]
        return p

    x_p = get_pred(x)
    xr_p = get_pred(xr)

    loss = T.log(T.sigm(x_p - xr_p))

    # add regularization terms
    reg = 0
    for w in Ws:
        reg += lambd * abs(w).mean()
    for b in bs:
        reg += lambd * abs(b).mean()

    return x, xr, Ws, bs, loss, reg #, x_p, xp_p, xr_p


def train():
    X_train, X_test, Xp_train, Xp_test, Xr_train, Xr_test, M_train, M_test, Y_train, Y_test = get_data()
    n_in = len(X_train[0])

    x, xr, Ws, bs, loss_f, reg_f = get_model(n_in)

    obj_f = loss_f + reg_f
    
    momentum = 0.9
    minibatch_size = min(10000, len(X_train))

    learning_rate = 1e-1
    while True:
        print 'learning rate:', learning_rate

        updates = []
        for param in Ws + bs:
            param_update = theano.shared(numpy.zeros(param.get_value().shape))
            updates.append((param, param - learning_rate * param_update))
            updates.append((param_update, momentum * param_update + (1. - momentum) * T.grad(obj_f, param)))

        # Compile
        train = theano.function(
            inputs=[x, xr],
            outputs=[loss_f, reg_f],
            updates=updates,
            on_unused_input='warn')

        test = theano.function(
            inputs=[x, xr],
            outputs=loss_f,
            on_unused_input='warn')

        # Train
        best_test_loss = float('inf')
        best_i = None

        for i in itertools.count():
            minibatch_index = random.randint(0, int(len(X_train) / minibatch_size) - 1)
            lo, hi = minibatch_index * minibatch_size, (minibatch_index + 1) * minibatch_size
            loss, reg = train(X_train[lo:hi], Xr_train[lo:hi])
            xs = [loss, reg]
            xs.append(sum(xs))

            test_loss = test(X_test, Xr_test)
            print '\t'.join(['%12.9f' % x for x in xs]) + ' test %12.9f' % test_loss

            # print '%5d: train obj: %5f (loss: %5f + reg: %5f), test loss: %5f' % (i, train_loss + train_reg, train_loss, train_reg, test_loss)

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_i = i

            if i > best_i + 400:
                break
        
            if (i+1) % 100 == 0:
                print 'dumping pickled model'
                f = open('model.pickle', 'w')
                def values(xs):
                    return [x.get_value(borrow=True) for x in xs]
                pickle.dump((values(Ws), values(bs)), f)
                f.close()

        learning_rate *= 0.5

if __name__ == '__main__':
    train()
