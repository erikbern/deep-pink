import train
import pickle
import theano
import theano.tensor as T
import math
import chess, chess.pgn
from parse_game import bb2array
import heapq
import time

f = open('model_0.pickle')
Ws, bs = pickle.load(f)

print Ws
print bs

Ws_s, bs_s = train.get_parameters(Ws=Ws, bs=bs)
x, p = train.get_model(Ws_s, bs_s)

predict = theano.function(
    inputs=[x],
    outputs=p)

gn_current = chess.pgn.Game()

class Node(object):
    def __init__(self, gn=None, score=None):
        self.gn = gn
        self.children = []
        self.score = score

while True:
    # Keep a heap of the most probable games
    n_root = Node(gn=gn_current)
    heap = []
    heap.append((0.0, n_root))

    sum_pos = 0.0

    # Do mini-max but evaluate all positions in the order of probability
    t0 = time.time()
    while time.time() - t0 < 10.0:
        neg_ll, n_current = heapq.heappop(heap)
        sum_pos += math.exp(-neg_ll)
        print sum_pos

        gn_candidates = []
        X = []
        for move in n_current.gn.board().legal_moves:
            gn_candidate = chess.pgn.GameNode()
            gn_candidate.parent = n_current.gn
            gn_candidate.move = move
            gn_candidates.append(gn_candidate)
            b = gn_candidate.board()
            flip = bool(b.turn == 0)
            X.append(bb2array(b, flip=flip))

        # Use model to predict scores
        scores = predict(X)

        # print 'inserting scores into heap'
        scores -= max(scores)
        log_z = math.log(sum([math.exp(s) for s in scores]))
        scores -= log_z

        for gn_candidate, score in zip(gn_candidates, scores):
            # print 'potential board w score', score
            # print gn_candidate.board()
            n_candidate = Node(gn_candidate, score)
            n_current.children.append(n_candidate)
            heapq.heappush(heap, (neg_ll - score, n_candidate))

    def evaluate(n):
        if n.gn.board().turn == 1:
            score = n.score
            f = -1
        else:
            score = None
            f = 1

        best_child = None

        if n.children:
            for n_child in n.children:
                score_child, _ = evaluate(n_child)
                if score_child:
                    if score is None or (score_child * f > score * f):
                        score = score_child
                        best_child = n_child

        return score, best_child

    score, best_child = evaluate(n_root)
    print 'score:', score
    gn_current = best_child.gn
    bb = gn_current.board()

    print bb

    def get_move(move_str):
        try:
            move = chess.Move.from_uci(move_str)
        except:
            print 'cant parse'
            return False
        if move not in bb.legal_moves:
            print 'not a legal move'
            return False
        else:
            return move

    while True:
        print 'your turn:'
        move = get_move(raw_input())
        if move:
            break

    gn_new = chess.pgn.GameNode()
    gn_new.parent = gn_current
    gn_new.move = move

    print gn_new.board()
    gn_current = gn_new

