import train
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


def negamax(gn_current, depth, alpha, beta, color, func):
    b = gn_current.board()
    # print depth, alpha, beta, color
    # print b

    if b.is_checkmate():
        return float('-inf'), None
    elif b.is_stalemate():
        return 0.0, None

    gn_candidates = []
    X = []
    for move in gn_current.board().legal_moves:
        gn_candidate = chess.pgn.GameNode()
        gn_candidate.parent = gn_current
        gn_candidate.move = move
        gn_candidates.append(gn_candidate)
        b = gn_candidate.board()
        flip = bool(b.turn == 0)
        X.append(bb2array(b, flip=flip))

    if len(X) == 0:
        raise Exception('eh?')
        # TODO: should treat checkmate

    # Use model to predict scores
    scores = func(X)

    child_nodes = sorted(zip(scores, gn_candidates), reverse=(color==1))

    best_value = float('-inf')
    best_move = None
    
    for score, gn_candidate in child_nodes:
        if depth == 1:
            value = score # * color
        else:
            neg_value, _ = negamax(gn_candidate, depth-1, -beta, -alpha, -color, func)
            value = -neg_value

        if value > best_value:
            best_value = value
            best_move = gn_candidate.move

        if value > alpha:
            alpha = value

        if alpha > beta:
            break

    return best_value, best_move


class Player(object):
    def move(self, gn_current):
        raise NotImplementedError()


class Computer(Player):
    def __init__(self, func):
        self._func = func

    def move(self, gn_current):
        assert(gn_current.board().turn == 0)

        for depth in xrange(1, 5):
            alpha = float('-inf')
            beta = float('inf')
            
            t0 = time.time()
            best_value, best_move = negamax(gn_current, depth, alpha, beta, 1, self._func)
            print depth, best_value, best_move, time.time() - t0
            depth += 1

        gn_new = chess.pgn.GameNode()
        gn_new.parent = gn_current
        gn_new.move = best_move

        return gn_new


class Human(Player):
    def move(self, gn_current):
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
        
        return gn_new


class Sunfish(Player):
    def __init__(self):
        import sunfish
        self._pos = sunfish.Position(sunfish.initial, 0, (True,True), (True,True), 0, 0)

    def move(self, gn_current):
        import sunfish

        assert(gn_current.board().turn == 1)

        # Apply last_move
        crdn = str(gn_current.move)
        move = (sunfish.parse(crdn[0:2]), sunfish.parse(crdn[2:4]))
        self._pos = self._pos.move(move)            

        move, score = sunfish.search(self._pos)
        self._pos = self._pos.move(move)

        crdn = sunfish.render(119-move[0]) + sunfish.render(119 - move[1])
        print crdn
        move = chess.Move.from_uci(crdn)
        
        gn_new = chess.pgn.GameNode()
        gn_new.parent = gn_current
        gn_new.move = move

        return gn_new

def play():
    func = get_model_from_pickle('model.pickle')

    gn_current = chess.pgn.Game()

    player_a = Computer(func)
    player_b = Sunfish()

    while True:
        gn_current = player_a.move(gn_current)
        print '=========== Player A:', gn_current.move
        print gn_current.board()
        gn_current = player_b.move(gn_current)
        print '=========== Player B:', gn_current.move
        print gn_current.board()

        
if __name__ == '__main__':
    play()
