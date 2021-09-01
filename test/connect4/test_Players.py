import numpy as np
import yaml

from connect4.GameEngine import Game, Board
from connect4.costants import *

from connect4.Players import FixedMovesPlayer

def test_FixedMovesPlayer():
    fm = list(range(N_COL))
    p = FixedMovesPlayer(fm)
    
    for i, m in enumerate(fm):
        assert(p.move(None, None, None) == fm[i])
    
    assert(p._index_of_next_move == N_COL)

    p.reset()
    assert(p._index_of_next_move == 0)

    for _ in range(N_COL):
        p.move(None, None, None)

    try:
        p.move(None, None, None)
    except IndexError:
        pass
    else:
        assert(False)

from connect4.Players import RandomPlayer

def test_RandomPlayer():
    # match
    g = Game(RandomPlayer(), RandomPlayer())
    g.play_all()
    
    # test: probabilities
    p = RandomPlayer()
    b = Board()
    # test probabilities: empty
    prob = p.probabilities(b, None, None)
    np.testing.assert_almost_equal(prob, np.ones(len(prob))/float(len(prob)))

    # test probabilities: 2 filled columns
    moves = [0]*N_COL + [1]*N_COL
    b.add_moves(moves, P1)
    prob = p.probabilities(b, moves, P1)
    goal_prob = np.ones(len(prob))
    goal_prob[0] = 0.
    goal_prob[1] = 0.
    goal_prob /= sum(goal_prob)
    np.testing.assert_almost_equal(prob, goal_prob)


from connect4.Players import MiniMaxPlayer

def test_MiniMaxPlayer():
    # match with show_scores
    g = Game(MiniMaxPlayer(2), MiniMaxPlayer(3, show_scores=True))
    g.play_all()


from connect4.Players import AlphaBetaPlayer

def test_AlphaBetaPlayer():
    # match
    g = Game(AlphaBetaPlayer(2), AlphaBetaPlayer(3))
    g.play_all()


from connect4.Players import CenteredAlphaBetaPlayer

def test_CenteredAlphaBetaPlayer():
    # match
    g = Game(CenteredAlphaBetaPlayer(2), CenteredAlphaBetaPlayer(3))
    g.play_all()


from connect4.Players import NoisyAlphaBetaPlayer

def test_NoisyAlphaBetaPlayer():
    # match
    g = Game(NoisyAlphaBetaPlayer(2, 1.), NoisyAlphaBetaPlayer(3, 1.))
    g.play_all()

    # impossible move, there is no score on which add the noise
    NoisyAlphaBetaPlayer(2, 1.).get_opponent_scores_given_move([0]*N_COL, P1)


from connect4.Players import PerfectPlayer

def get_perfect_player_path():
    with open('test/connect4/conf/perfect-player-path.yaml') as ppp_file:
        contenent = yaml.safe_load(ppp_file)
        return contenent['perfect-player-path']

def test_PerfectPlayer():
    perfect_player_path = get_perfect_player_path()
    g = Game(
        PerfectPlayer(perfect_player_path, show_scores=True),
        PerfectPlayer(perfect_player_path)
    )
    g.play_all()
    assert(g.winner == P1)


from connect4.Players import TensorFlowProabilitiesPlayer

def get_tf_probabilities_path():
    with open('test/connect4/conf/tf-probabilities-path.yaml') as tpp_file:
        contenent = yaml.safe_load(tpp_file)
        return contenent['tf-probabilities-path']

def test_TensorFlowProabilitiesPlayer():
    tf_probabilities_path = get_tf_probabilities_path()
    g = Game(
        TensorFlowProabilitiesPlayer(tf_probabilities_path),
        TensorFlowProabilitiesPlayer(tf_probabilities_path),
    )
    g.play_all()


from connect4.Players import TensorFlowScorePlayer

def get_tf_score_path():
    with open('test/connect4/conf/tf-score-path.yaml') as tsp_file:
        contenent = yaml.safe_load(tsp_file)
        return contenent['tf-score-path']

def test_TensorFlowScorePlayer():
    tf_score_path = get_tf_score_path()
    g = Game(
        TensorFlowScorePlayer(tf_score_path),
        TensorFlowScorePlayer(tf_score_path),
    )
    g.play_all()


from connect4.Players import TwoStagePlayer

def test_TwoStagePlayer():
    tf_score_path = get_tf_score_path()
    g = Game(
        TwoStagePlayer(RandomPlayer(), AlphaBetaPlayer(2), 3),
        TwoStagePlayer(RandomPlayer(), CenteredAlphaBetaPlayer(4), 5),
    )
    g.play_all()

from connect4.Players import PoolPlayer

def test_PoolPlayer():
    g = Game(
        PoolPlayer([RandomPlayer(), RandomPlayer(), CenteredAlphaBetaPlayer(2)]),
        PoolPlayer([RandomPlayer(), RandomPlayer(), CenteredAlphaBetaPlayer(2)]),
    )
    g.play_all()


# Playing Tournament requires that every Player has reset() and terminate() methods
# correctly implemented. I'd like to test them all. Exceptions are:
#  - ConsolePlayer: can be tested easilly with pytest;
from connect4.GameEngine import Tournament

def test_all_players_with_Tournament():
    # AlphaBeta & CenteredAlphaBeta
    t = Tournament(
        AlphaBetaPlayer(2),
        CenteredAlphaBetaPlayer(2)
    )
    t.play_games(2)

    # FixedMoves
    t = Tournament(
        FixedMovesPlayer([0, 1, 2, 2, 3, 3]),
        FixedMovesPlayer([1, 2, 3, 3, 0, 1])
    )
    t.play_games(2)
    assert(t.counter[P1][P1] == t.counter[P2][P1])

    # Perfect & NoisyAlphaBeta
    t = Tournament(
        PerfectPlayer(get_perfect_player_path()),
        NoisyAlphaBetaPlayer(3, 0.)
    )
    t.play_games(2)
    assert(t.counter[P1][P1] == t.counter[P2][P1])

    # TwoStagePlayer & PoolPlayer
    t = Tournament(
        TwoStagePlayer(RandomPlayer(), CenteredAlphaBetaPlayer(4), 1),
        PoolPlayer([RandomPlayer(), RandomPlayer(), CenteredAlphaBetaPlayer(2)])
    )
    t.play_games(2)
    assert(t.counter[P1][P1] == t.counter[P2][P1])

