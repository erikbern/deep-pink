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
    X_data = []
    y_data = []

    def read_stuff():
        for fn in os.listdir(dir):
            if not fn.endswith('.npy'):
                continue

            fn = os.path.join(dir, fn)
            print fn
            f = open(fn)

            while True:
                try:
                    y = numpy.load(f)
                    x = numpy.load(f)
                except:
                    break
                
                yield x, y

    for x, y in read_stuff():
        y_data.append(y)
        X_data.append(x)

        if len(y_data) % 1000 == 0:
            print len(y_data), '...'

        if len(y_data) == 1000000:
            break

    return X_data, y_data


def get_data():
    X_data, y_data = load_data()

    print 'read', len(X_data), 'positions'
    
    X_data = numpy.array(X_data)
    y_data = numpy.array(y_data)

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.01)

    print len(X_train), 'train set', len(X_test), 'test set'

    return X_train, X_test, y_train, y_test

def get_model(n_in):
    n_hidden_layers = 4
    n_hidden = 2048

    # Declare Theano symbolic variables
    x = T.matrix("x")
    y = T.vector("y")

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

    return (x, y, Ws, bs)

def build_network(x, y, Ws, bs, dropout=False, lambd=1.0):
    # Construct Theano expression graph
    srng = theano.tensor.shared_randomstreams.RandomStreams(
        rng.randint(999999))

    last_layer = x
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

    p = T.tanh(T.dot(last_layer, Ws[-1]) + bs[-1])
    xent = (y - p) ** 2
    loss = xent.mean()

    # add regularization terms
    reg = 0
    for w in Ws:
        reg += lambd * abs(w).mean()
    for b in bs:
        reg += lambd * abs(b).mean()
    return p, loss, reg


def train():
    X_train, X_test, y_train, y_test = get_data()
    n_in = len(X_train[0])

    x, y, Ws, bs = get_model(n_in)

    _, train_loss_f, train_reg_f = build_network(x, y, Ws, bs, dropout=False)
    _, test_loss_f, _ = build_network(x, y, Ws, bs, dropout=False)

    train_obj_f = train_loss_f + train_reg_f
    
    momentum = 0.9
    minibatch_size = min(10000, len(X_train))

    learning_rate = 0.1
    while True:
        print 'learning rate:', learning_rate

        updates = []
        for param in Ws + bs:
            param_update = theano.shared(numpy.zeros(param.get_value().shape))
            updates.append((param, param - learning_rate * param_update))
            updates.append((param_update, momentum * param_update + (1. - momentum) * T.grad(train_obj_f, param)))

        # Compile
        train = theano.function(
            inputs=[x,y],
            outputs=[train_obj_f, train_loss_f, train_reg_f],
            updates=updates)
        test = theano.function(
            inputs=[x,y],
            outputs=test_loss_f)

        # Train
        best_test_loss = float('inf')
        best_i = None

        for i in itertools.count():
            minibatch_index = random.randint(0, int(len(X_train) / minibatch_size) - 1)
            lo, hi = minibatch_index * minibatch_size, (minibatch_index + 1) * minibatch_size
            train_obj, train_loss, train_reg = train(X_train[lo:hi], y_train[lo:hi])
            test_loss = test(X_test, y_test)
            print '%5d: train obj: %5f (loss: %5f + reg: %5f), test loss: %5f' % (i, train_obj, train_loss, train_reg, test_loss)

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
