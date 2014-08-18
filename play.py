import train
import pickle
import theano
import theano.tensor as T
import math

f = open('model.pickle')
Ws, bs = pickle.load(f)

print Ws
print bs

Ws_s, bs_s = train.get_parameters(Ws=Ws, bs=bs)
x, p = train.get_model(Ws_s, bs_s)

predict = theano.function(
    inputs=[x],
    outputs=p)

import chess

bb = chess.Bitboard()

from parse_game import bb2array

while True:
    best_move = None
    best_score = float('-inf')
    scores = []
    for move in bb.legal_moves:
        bb.push(move)
        x = bb2array(bb)
        score, = predict([x])

        scores.append(score)

        if score > best_score:
            best_move, best_score = move, score
            
        bb.pop()

    # calculate softmax probability
    m = max(scores)
    Z = sum([math.exp(s - m) for s in scores])

    print 'best move', best_move, 'score', best_score, 'prob', math.exp(best_score - m) / Z
    if best_move is None:
        break
    bb.push(best_move)

    print bb

    print 'your turn:'
    move = raw_input()

    bb.push(chess.Move.from_uci(move))

    print bb

