import chess, chess.pgn
import numpy
import sys
import os
import multiprocessing
import itertools
import random

def read_games(fn):
    f = open(fn)

    while True:
        try:
            g = chess.pgn.read_game(f)
        except KeyboardInterrupt:
            raise
        except:
            continue

        if not g:
            break
        
        yield g


def bb2array(b, flip=False):
    x = numpy.zeros((2, 6, 64), dtype=numpy.int8)
    
    for pos in xrange(64):
        p = b.piece_at(pos)
        if p is not None:
            col = int(pos % 8)
            row = int(pos / 8)
            if flip:
                row = 7-row
                color = 1 - p.color
            else:
                color = p.color
            x[p.color][p.piece_type - 1][row * 8 + col] = 1

    return x.flatten()


def parse_game(g):
    rm = {'1-0': 1, '0-1': -1, '1/2-1/2': 0}
    r = g.headers['Result']
    if r not in rm:
        return None
    y = rm[r]
    # print >> sys.stderr, 'result:', y

    # Generate all boards
    gn = g.end()
    if not gn.board().is_game_over():
        return None

    gns = []
    moves_left = 0
    while gn:
        gns.append((moves_left, gn, gn.board().turn == 0))
        gn = gn.parent
        moves_left += 1

    print len(gns)
    if len(gns) < 10:
        print g.end()

    gns.pop()

    moves_left, gn, flip = random.choice(gns)

    x = bb2array(gn.board(), flip=flip)
    b_parent = gn.parent.board()
    x_parent = bb2array(gn.parent.board(), flip=(not flip))

    # generate a random baord
    moves = list(b_parent.legal_moves)
    move = random.choice(moves)
    b_parent.push(move)
    x_random = bb2array(b_parent, flip=flip)

    
    # print x
    # print x_parent
    # print x_random

    return (x, x_parent, x_random, moves_left, y)


def read_all_games(fn_in, fn_out):
    g = open(fn_out, 'w')
    pool = multiprocessing.Pool()
    for game in itertools.imap(parse_game, read_games(fn_in)): # pool.imap_unordered(parse_game, read_games(fn_in), chunksize=100):
        if game is None:
            continue
        x, x_parent, x_random, moves_left, y = game
        numpy.save(g, x.flatten())
        numpy.save(g, x_parent.flatten())
        numpy.save(g, x_random.flatten())
        numpy.save(g, moves_left)
        numpy.save(g, y)

    g.close()

if __name__ == '__main__':
    d = '/mnt/games'
    for fn_in in os.listdir(d):
        if not fn_in.endswith('.pgn'):
            continue
        fn_in = os.path.join(d, fn_in)
        fn_out = fn_in.replace('.pgn', '_2.npy')
        if not os.path.exists(fn_out):
            print 'reading', fn_in
            read_all_games(fn_in, fn_out)
