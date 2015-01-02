import train
import theano
import theano.tensor as T
import chess, chess.pgn
from parse_game import bb2array
import numpy
import random
import pickle
from theano.tensor.nnet import sigmoid


def get_params(fn):
    f = open(fn)
    Ws, bs = pickle.load(f)
    return Ws, bs


def get_predict(Ws, bs):
    Ws_s, bs_s = train.get_parameters(Ws=Ws, bs=bs)
    x, p = train.get_model(Ws_s, bs_s)
    
    predict = theano.function(
        inputs=[x],
        outputs=p)

    return predict


def get_update(Ws, bs, learning_rate=1e-6, momentum=0.9):
    Ws_s, bs_s = train.get_parameters(Ws=Ws, bs=bs)
    x, fx = train.get_model(Ws_s, bs_s)

    # Ground truth (who won)
    y = T.vector('y')

    # Compute loss (just log likelihood of a sigmoid fit)
    y_pred = sigmoid(fx)
    loss = -( y * T.log(y_pred) + (1 - y) * T.log(1 - y_pred)).mean()

    # Updates
    updates = train.nesterov_updates(loss, Ws_s + bs_s, learning_rate, momentum)
    
    f_update = theano.function(
        inputs=[x, y],
        outputs=loss,
        updates=updates,
        )

    return f_update


def weighted_random_sample(ps):
    r = random.random()
    for i, p in enumerate(ps):
        r -= p
        if r < 0:
            return i


def game(f_pred, f_train):
    print 'new game...'
    gn = chess.pgn.Game()
    b = gn.board()

    data = []

    while not b.is_game_over():
        # Generate all possible moves
        Xs = []
        gns = []
        for move in b.legal_moves:
            gn_new = chess.pgn.GameNode()
            gn_new.parent = gn
            gn_new.move = move
            Xs.append(bb2array(gn_new.board(), flip=b.turn))
            gns.append(gn_new)

        # Calculate softmax probabilities
        ys = f_pred(Xs)
        zs = numpy.exp(ys)
        Z = sum(zs)
        ps = zs / Z
        i = weighted_random_sample(ps)

        # Append moves
        data.append((b.turn, Xs[i]))

        gn = gns[i]
        b = gn.board()
    
    if not b.is_checkmate():
        return

    X = numpy.array([x for t, x in data], dtype=theano.config.floatX)
    Y = numpy.array([(t ^ b.turn) for t, x in data], dtype=theano.config.floatX)

    f_train(X, Y)

def main():
    Ws, bs = get_params('model.pickle')
    f_pred = get_predict(Ws, bs)
    f_train = get_update(Ws, bs)

    while True:
        game(f_pred, f_train)

if __name__ == '__main__':
    main()
