import train, train_y
import pickle
import theano
import theano.tensor as T
import math
import chess, chess.pgn
from parse_game import bb2array
import heapq
import time

def get_model_from_pickle(fn):
    f = open(fn)
    Ws, bs = pickle.load(f)
    
    Ws_s, bs_s = train.get_parameters(Ws=Ws, bs=bs)
    x, p = train.get_model(Ws_s, bs_s)
    
    predict = theano.function(
        inputs=[x],
        outputs=p)

    return predict

move_func = get_model_from_pickle('model.pickle')
eval_func = get_model_from_pickle('model_y.pickle')

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
        print sum_pos, len(heap)

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

        if len(X) == 0:
            # TODO: should treat checkmate
            continue

        # Use model to predict scores
        move_scores = move_func(X)
        eval_scores = eval_func(X)

        move_scores *= 0.5 # some smoothing heuristic to make it less confident

        # print 'inserting scores into heap'
        move_scores -= max(move_scores)
        log_z = math.log(sum([math.exp(s) for s in move_scores]))
        move_scores -= log_z

        for gn_candidate, move_score, eval_score in zip(gn_candidates, move_scores, eval_scores):
            n_candidate = Node(gn_candidate, eval_score)
            n_current.children.append(n_candidate)
            heapq.heappush(heap, (neg_ll - move_score, n_candidate))


    def evaluate(n, level=0):
        if n.gn.board().turn == 1:
            f = -1
        else:
            f = 1

        score = None
        n.best_child = None

        if n.children:
            for n_child in n.children:
                score_child, _ = evaluate(n_child, level+1)
                if score_child:
                    if score is None or (score_child * f > score * f):
                        score = score_child
                        n.best_child = n_child

        if score is None:
            # Use leaf value
            score = n.score

        if level < 3:
            print '\t' * level, score, n.score, f, n.gn.move

        return score, n.best_child

    print 'performing minimax'
    score, best_child = evaluate(n_root)
    print 'score:', score
    print 'most likely event of moves'
    n = n_root
    while n is not None:
        print n.score
        print n.gn.board()
        print
        n = n.best_child
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

