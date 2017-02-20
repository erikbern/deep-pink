import train
import load
import theano
import theano.tensor as T
# import chess, chess.pgn
from parse_game import bb2array
import numpy
import random
import pickle
from theano.tensor.nnet import sigmoid
import sunfish
from play import sf2array
import os
import time
import math


def dump(Ws_s, bs_s):
    f = open('model_reinforcement.pickle', 'w')
    def values(zs):
        return [z.get_value(borrow=True) for z in zs]
    pickle.dump((values(Ws_s), values(bs_s)), f)


def get_params(fns):
    for fn in fns:
        if os.path.exists(fn):
            print 'loading', fn
            f = open(fn)
            Ws, bs = pickle.load(f)
            return Ws, bs


def get_predict(Ws_s, bs_s):
    x, p = load.get_model(Ws_s, bs_s)
    
    predict = theano.function(
        inputs=[x],
        outputs=p)

    return predict


def get_update(Ws_s, bs_s):
    x, fx = load.get_model(Ws_s, bs_s)

    # Ground truth (who won)
    y = T.vector('y')

    # Compute loss (just log likelihood of a sigmoid fit)
    y_pred = sigmoid(fx)
    loss = -( y * T.log(y_pred) + (1 - y) * T.log(1 - y_pred)).mean()

    # Metrics on the number of correctly predicted ones
    frac_correct = ((fx > 0) * y + (fx < 0) * (1 - y)).mean()

    # Updates
    learning_rate_s = T.scalar(dtype=theano.config.floatX)
    momentum_s = T.scalar(dtype=theano.config.floatX)
    updates = train.nesterov_updates(loss, Ws_s + bs_s, learning_rate_s, momentum_s)
    
    f_update = theano.function(
        inputs=[x, y, learning_rate_s, momentum_s],
        outputs=[loss, frac_correct],
        updates=updates,
        )

    return f_update


def weighted_random_sample(ps):
    r = random.random()
    for i, p in enumerate(ps):
        r -= p
        if r < 0:
            return i


def game(f_pred, f_train, learning_rate, momentum=0.9):
    pos = sunfish.Position(sunfish.initial, 0, (True,True), (True,True), 0, 0)

    data = []

    max_turns = 100

    for turn in xrange(max_turns):
        # Generate all possible moves
        Xs = []
        new_poss = []
        for move in pos.genMoves():
            new_pos = pos.move(move)
            Xs.append(sf2array(new_pos, False))
            new_poss.append(new_pos)

        # Calculate softmax probabilities
        ys = f_pred(Xs)
        zs = numpy.exp(ys)
        Z = sum(zs)
        ps = zs / Z
        i = weighted_random_sample(ps)

        # Append moves
        data.append((turn % 2, Xs[i]))
        pos = new_poss[i]

        if pos.board.find('K') == -1:
            break

        if turn == 0 and random.random() < 0.01:
            print ys

    if turn == max_turns - 1:
        return

    # White moves all even turns
    # If turn is even, it means white just moved, and black is up next
    # That means if turn is even, all even (black) boards are losses
    # If turn is odd, all odd (white) boards are losses
    win = (turn % 2) # 0 = white, 1 = black

    X = numpy.array([x for t, x in data], dtype=theano.config.floatX)
    Y = numpy.array([(t ^ win) for t, x in data], dtype=theano.config.floatX)

    loss, frac_correct = f_train(X, Y, learning_rate, momentum)

    return len(data), loss, frac_correct


def main():
    Ws, bs = get_params(['model_reinforcement.pickle', 'model.pickle'])
    Ws_s, bs_s = load.get_parameters(Ws=Ws, bs=bs)
    f_pred = get_predict(Ws_s, bs_s)
    f_train = get_update(Ws_s, bs_s)

    i, n, l, c = 0, 0.0, 0.0, 0.0

    base_learning_rate = 1e-2
    t0 = time.time()

    while True:
        learning_rate = base_learning_rate * math.exp(-(time.time() - t0) / 86400)
        r = game(f_pred, f_train, learning_rate)
        if r is None:
            continue
        i += 1
        n_t, l_t, c_t = r
        n = n*0.999 + n_t
        l = l*0.999 + l_t*n_t
        c = c*0.999 + c_t*n_t
        print '%6d %9.5f %9.5f %9.5f' % (i, learning_rate, l / n, c / n)

        if i % 100 == 0:
            print 'dumping model...'
            dump(Ws_s, bs_s)


if __name__ == '__main__':
    main()
