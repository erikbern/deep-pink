import numpy
import theano
import theano.tensor as T
rng = numpy.random

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
                gamma = 0.1 # initialize it to slightly positive so the derivative exists
                b = numpy.ones(n_out_2, dtype=theano.config.floatX) * gamma
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

    if type(dropout) != list:
        dropout = [dropout] * len(Ws_s)

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
        
        if dropout[l]:
            mask = srng.binomial(n=1, p=0.5, size=h.shape)
            h = h * T.cast(mask, theano.config.floatX) * 2

        last_layer = h

    p_s = T.dot(last_layer, Ws_s[-1]) + bs_s[-1]
    return x_s, p_s

