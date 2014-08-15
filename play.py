from train import build_network
import pickle
import theano
import theano.tensor as T

f = open('model.pickle')
Ws, bs = pickle.load(f)

print Ws
print bs

def make_theano(xs, symbol):
    return [theano.shared(x, name='%s%d' % (symbol, i)) for i, x in enumerate(xs)]

x = T.matrix("x")
y = T.vector("y")

Ws = make_theano(Ws, 'w')
bs = make_theano(bs, 'b')

p, _, _ = build_network(x, y, Ws, bs)

predict = theano.function(
    inputs=[x],
    outputs=p)

import chess

bb = chess.Bitboard()

from parse_game import bb2array

while True:
    best_move = None
    best_score = -1
    for move in bb.legal_moves:
        bb.push(move)
        x = bb2array(bb)
        score, = predict([x])

        if score > best_score:
            best_move, best_score = move, score
            
        bb.pop()

    print 'best move', best_move, 'score', best_score
    if best_move is None:
        break
    bb.push(best_move)

    print bb

    print 'your turn:'
    move = raw_input()

    bb.push(chess.Move.from_uci(move))

    print bb

