import numpy as np

from connect4.GameEngine import change_player, Board
from connect4.costants import *

def test_change_player():
    assert(change_player(P1) == P2)
    assert(change_player(P2) == P1)
    try:
        change_player(NOBODY)
    except ValueError:
        pass

def test_Board_init():
    b = Board()

    try:
        b = Board([])
    except ValueError:
        pass

    b = Board([0,1,2,3], P1)

def test_Board_empty():
    b = Board([0,1,2,3], P1)
    b.empty()
    for c in range(N_COL):
        assert(len(b.board[c]) == 0)

def test_Board_add_moves():
    b = Board(list(range(N_COL)), P1)
    for c in range(N_COL):
        if c % 2 == 0:
            assert(b.board[c][0] == P1 and len(b.board[c]) == 1)
        else:
            assert(b.board[c][0] == P2 and len(b.board[c]) == 1)

    b.add_moves(list(range(N_COL)), P2)
    for c in range(N_COL):
        if c % 2 == 0:
            assert(b.board[c][1] == P2 and len(b.board[c]) == 2)
        else:
            assert(b.board[c][1] == P1 and len(b.board[c]) == 2)    

def test_Board_add_move():
    b = Board()
    b.add_move(0, P1)

    for _ in range(N_ROW):
        b.add_move(0, P1)

def test_Board_is_allowed():
    b = Board([0 for _ in range(N_ROW)], P1)

    assert(b.is_allowed(1))
    assert(not b.is_allowed(0))

def test_Board_evalutate():
    def assert_evalutation(board, winner, points):
        if points is not None:
            winner, wp = board.evalutate(True)
        else:
            winner, wp = board.evalutate(False)

        assert(winner == winner)
        if points is not None:
            assert(wp == points)
        else:
            assert(wp == 0)

    # Board full of P1
    b = Board()
    for r in range(N_ROW):
        for c in range(N_COL):
            b.add_move(c, P1)
    assert_evalutation(b, P1, None)
    assert_evalutation(b, P1, N_ROW*N_COL)

    # Row P1
    b = Board()
    for c in range(4):
        b.add_move(c, P1)
    assert_evalutation(b, P1, None)
    assert_evalutation(b, P1, 4)

    # Col P2
    b = Board()
    for _ in range(4):
        b.add_move(2, P2)
    assert_evalutation(b, P2, None)
    assert_evalutation(b, P2, 4)
    
    # Diagonal Direct P2
    b = Board()
    for c in range(4):
        for r in range(c):
            b.add_move(c, P1)
        b.add_move(c, P2)
    assert_evalutation(b, P2, None)
    assert_evalutation(b, P2, 4)

    # Diagonal Inverse P1
    b = Board()
    for c in range(4):
        for r in range(3-c):
            b.add_move(c, P1)
        b.add_move(c, P2)
    assert_evalutation(b, P1, None)
    assert_evalutation(b, P1, 4)

    # Nobody
    b = Board()
    assert_evalutation(b, NOBODY, None)
    assert_evalutation(b, NOBODY, 0)

    # Not valid 
    b = Board([0]*(N_ROW + 1), P1)
    assert(b.evalutate() == (None, None))

def test_Board_as_numpy():
    # Empty
    b = Board()
    assert(np.array_equal(
        b.as_numpy(P1),
        np.zeros((N_ROW, N_COL), dtype=np.int8)
        )
    )

    # Two coins
    b = Board()
    b.add_move(N_COL-1, P1)
    b.add_move(0, P2)
    np_board = np.zeros((N_ROW, N_COL), dtype=np.int8)
    np_board[0][0] = 1
    np_board[0][N_COL-1] = -1
    assert(np.array_equal(
        b.as_numpy(P2),
        np_board
        )
    )

from connect4.GameEngine import Game
from connect4.Players import RandomPlayer, FixedMovesPlayer

def test_Game_init():
    g = Game(RandomPlayer(), RandomPlayer())

def test_Game_check_finish():
    g = Game(
        RandomPlayer(), 
        RandomPlayer()
    )
    g.play_all()
    assert(g.check_finish())

def test_Game_insert_coin():
    # Try to move in a forbidden column
    g = Game(
        FixedMovesPlayer([0]*int((N_ROW+1)/2)),
        FixedMovesPlayer([0]*int(N_ROW/2))
    )
    for _ in range(N_ROW):
        g.next()
    print(g)
    try:
        g.insert_coin(0, g.turn)
    except ValueError:
        pass
    else:
        assert(False)

def test_Game_next():
    # Move in an ended game
    g = Game(
        FixedMovesPlayer([0]*4),
        FixedMovesPlayer([1]*3)
    )
    g.play_all()
    g.next()

def test_Game_play_all():
    test_Game_check_finish()


from connect4.GameEngine import Tournament

def test_Tournament():
    # alternate
    t = Tournament(
        RandomPlayer(),
        RandomPlayer(),
        starting_criteria='alternate'
    )
    t.play_games(10)
    t.finished()
    assert(
        t.counter[P1][P1] + t.counter[P1][P2] + t.counter[P1][NOBODY] == 
        t.counter[P2][P1] + t.counter[P2][P2] + t.counter[P2][NOBODY]
    )

    # p1
    t = Tournament(
        RandomPlayer(),
        RandomPlayer(),
        starting_criteria='p1'
    )
    t.play_games(10)
    t.finished()
    assert(
        t.counter[P1][P1] + t.counter[P1][P2] + t.counter[P1][NOBODY] == 
        10
    )

    # p2
    t = Tournament(
        RandomPlayer(),
        RandomPlayer(),
        starting_criteria='p2'
    )
    t.play_games(10)
    t.finished()
    assert(
        t.counter[P2][P1] + t.counter[P2][P2] + t.counter[P2][NOBODY] == 
        10
    )

    # random
    t = Tournament(
        RandomPlayer(),
        RandomPlayer(),
        starting_criteria='random'
    )
    t.play_games(10)
    t.finished()
    assert(
        t.counter[P1][P1] + t.counter[P1][P2] + t.counter[P1][NOBODY] +
        t.counter[P2][P1] + t.counter[P2][P2] + t.counter[P2][NOBODY]
        == 10
    )

    # unknown starting
    try:
        t = Tournament(
            RandomPlayer(),
            RandomPlayer(),
            starting_criteria = 'nicolò_barella'
        )
        t.play_games(10)
        t.finished()
    except ValueError:
        pass
    else:
        assert(False)

        

    

    



    
