"""
Some players implementations for connect4.
"""

import random as rnd
import numpy as np
import tensorflow as tf
import subprocess
from copy import deepcopy
import operator
import multiprocessing as mp
from scipy.stats import mode

from .GameEngine import *
from .algorithms import minimax, alphabeta, WORSE_SCORE, MOVES_ORDER, MIN_SCORE_STEP, SCORE_TO_LOGPROB

class FixedMovesPlayer(Player):
    """
    Play a predefined sequence of moves.


    Attributes
    ----------
    name: `string`, default: 'FixedMoves Player'
        Name used to refer to the player.
    moves: `list(move)`
    """
    def __init__(self, moves, name = 'FixedMoves Player'):
        super().__init__(name)
        self.moves = moves
        self._index_of_next_move = 0

    def move(self, board, moves, self_player):
        """
        See :func:`GameEngine.Player.move` for arguments explenation.
        """
        m = self.moves[self._index_of_next_move]
        self._index_of_next_move += 1
        return m
    
    def reset(self):
        self._index_of_next_move = 0

class ConsolePlayer(Player): # pragma: no cover
    # Can't be tested easilly with pytest.
    """
    Play a game throught console.

    This is an interface for a player that takes the moves from the console.

    Attributes
    ----------
    name: `string`, default: 'Console Player'
        Name used to refer to the player.
    """
    def __init__(self, name = 'Console Player'):
        super().__init__(name)

    def move(self, board, moves, self_player, print_board = True):
        """
        stdin input for next move.

        The move is to be inserted in the console is a number from 1 ro N_COL,
        from left to right.
        Argouments are unused.
        See :func:`GameEngine.Player.move` for arguments explenation.
        """
        if print_board:
            print(board)
        return int(input('Which column? '))-1

class RandomPlayer(Player):
    """
    Random Player.

    This player choose randomly (uniform) which allowed move do.

    Attributes
    ----------
    name: `string`, default: 'Random Player'
        Name used to refer to the player.
    """
    def __init__(self, name = 'Random Player'):
        super().__init__(name)

    def move(self, board, moves, self_player):
        """
        See :func:`GameEngine.Player.move` for arguments explenation.
        """
        while True:
            c = rnd.randint(0, N_COL-1)
            if len(board.board[c]) < N_ROW:
                return c
    
    def probabilities(self, board, moves, self_player):
        p = np.zeros((N_COL,))
        for c in range(N_COL):
            if len(board.board[c]) < N_ROW:
                p[c] = 1.
        return p/sum(p)

class MiniMaxPlayer(Player):
    """
    Play a game using MiniMax algorithm implemented in :func:`algorithms.minimax`.

    Attributes
    ----------
    name: `string`, default: 'MiniMax Player'
        Name used to refer to the player.
    depth: `int`
        Depth of the tree exploration; it represent how many moves in advance the
        algorithm look.
    show_scores: `bool`, default: `False`
        If it is `True` the scores  of each moves are printed on `stdout` when
        method :func:`~Players.MiniMaxPlayer.move` is called.
    """
    def __init__(self, depth, show_scores = False, name = 'MiniMax Player'):
        super().__init__(name)
        self.depth = depth
        self.show_scores = show_scores

    def get_opponent_scores_given_move(self, moves, self_player):
        """
        Get the scores best possible scores for the opponent, for each possible move
        allowed in this situation. The algorithm :func:`algorithms.minimax` is used.

        Parameters
        ----------
        moves: `list`
            List of moves that  have been done throught the game.
            A move is a number in interval ``[0,N_COL)``, indicating the
            chosen column from left to right.
            Moves do not distinguish wich player did it.
        self_player: `player_tag`
            Tag of the current player.
        Returns
        -------
        `list(float and/or NoneType)`
            A list of `lenght = N_COL`, where the element in position `i`
            is the opponent score if the player do move `i`.
            If move `i` is not allowed there's a `None` instead of a float.
        """
        return [minimax(moves + [m], self.depth-1) for m in range(N_COL)]

    def _remove_None_from_scores(self, scores):
        for m in range(N_COL):
            try:
                scores[m] *= -1
            except TypeError:
                scores[m] = -float('inf')

    def move(self, board, moves, self_player):
        """
        See :func:`GameEngine.Player.move` for arguments explenation.
        """
        scores = self.get_opponent_scores_given_move(moves, self_player)
        self._remove_None_from_scores(scores)
        
        if self.show_scores:
            prob = self.probabilities(board, moves, self_player)
            for m in range(N_COL):
                print(f'{m+1}: {scores[m]:.1f}, prob = {prob[m]:.3f}')

        return max(range(N_COL), key = lambda i: scores[i] if scores[i] is not None else 2*WORSE_SCORE)

    def probabilities(self, board, moves, self_player):
        """
        See :func:`GameEngine.Player.probabilities` for arguments explenation.
        """
        scores = self.get_opponent_scores_given_move(moves, self_player)
        self._remove_None_from_scores(scores)

        p = np.exp(np.array(scores)*SCORE_TO_LOGPROB)
        return p/sum(p)

class AlphaBetaPlayer(MiniMaxPlayer):
    """
    Play a game using MiniMax algorithm with alpha-beta pruning implemented in :func:`algorithms.alphabeta`.

    Attributes
    ----------
    name: `string`, default: 'Alpha-Beta Player'
        Name used to refer to the player.
    depth: `int`
        Depth of the tree exploration; it represent how many moves in advance the
        algorithm look.
    show_scores: `bool`, default: `False`
        If it is `True` the scores  of each moves are printed on `stdout` when
        method :func:`~Players.MiniMaxPlayer.move` is called.
    """
    def __init__(self, depth, show_scores = False, name = 'Alpha-Beta Player'):
        super().__init__(depth, show_scores, name)

    def get_opponent_scores_given_move(self, moves, self_player):
        """
        Get the scores best possible scores for the opponent, for each possible move
        allowed in this situation. The algorithm :func:`algorithms.alphabeta` is used.

        See :func:`Players.MiniMaxPlayer.get_opponent_scores_given_move` for arguments explenation.
        """
        return [alphabeta(moves = moves + [m],
                            alpha = WORSE_SCORE,
                            beta = -WORSE_SCORE,
                            depth = self.depth-1)
                for m in range(N_COL)]

class CenteredAlphaBetaPlayer(AlphaBetaPlayer):
    """
    It is a player identical to :class:`Players.AlphaBetaPlayer`, except that the
    criteria for breaking scores ties is given by `costants.MOVE_ORDER`. This heuristic increase
    the probability of exploring the subtree of the best move first, increasing significantly
    the performance of alpha-beta algorithm.
    """
    def get_opponent_scores_given_move(self, moves, self_player):
        scores = super().get_opponent_scores_given_move(moves, self_player)
        for i in range(len(MOVES_ORDER)):
            try:
                scores[MOVES_ORDER[i]] += i * MIN_SCORE_STEP / N_COL / 10
            except TypeError:
                pass
        return scores

class NoisyAlphaBetaPlayer(AlphaBetaPlayer):
    """
    Overlap a Gaussian Noise to :class:`Players.AlphaBetaPlayer` score to not be deterministic.

    Attributes
    ----------
    name: `string`, default: 'NoisyAlphaBeta Player'
        Name used to refer to the player.
    depth: `int`
        Depth of the tree exploration; it represent how many moves in advance the
        algorithm look.
    show_scores: `bool`, default: `False`
        If it is `True` the scores  of each moves are printed on `stdout` when
        method :func:`~Players.MiniMaxPlayer.move` is called.
    noise: `float`
        Standard deviation of the gaussian noise.
    """
    def __init__(self, depth, noise, show_scores = False, name = 'NoisyAlphaBeta Player'):
        super().__init__(depth, show_scores, name)
        self.noise = noise

    def get_opponent_scores_given_move(self, moves, self_player):
        scores = super().get_opponent_scores_given_move(moves, self_player)
        for m in MOVES_ORDER:
            try:
                scores[m] += np.random.normal(0., self.noise)
            except TypeError:
                pass
        return scores

class PerfectPlayer(Player):
    """
    Player that always do the best move in each configuration.

    Attributes
    ----------
    name: `string`, default: 'Perfect Player'
        Name used to refer to the player.
    oracle: `subprocess.Popen`
        Subprocess to interact with for getting the best move.
    show_scores: `bool`, default: `False`
        If it is `True` the scores  of each moves are printed on `stdout` when
        method :func:`~Players.MiniMaxPlayer.move` is called.
    """
    def __init__(self, executable_path, show_scores = False, name = 'Perfect Player'):
        """
        Parameters
        ----------
        executable_path: `string`
            Path to the executable file which runs the perfect player.
        """
        super().__init__(name)

        self.oracle = subprocess.Popen([executable_path, '-a'],
                                        stdout = subprocess.PIPE,
                                        stdin = subprocess.PIPE,
                                        stderr = subprocess.PIPE)
        self.oracle.stderr.readline()
        self.show_scores = show_scores

    def finished(self):
        """
        Kill `oracle` process.

        See :func:`GameEngine.Player.finished`
        """
        self.oracle.terminate()

    def move(self, board, moves, self_player):
        """
        See :func:`GameEngine.Player.move` for arguments explenation.
        """
        moves_string = ''
        for m in moves:
            moves_string += f'{m+1}'

        self.oracle.stdin.write(str.encode(f'{moves_string}\n'))
        self.oracle.stdin.flush()
        oracle_answer = self.oracle.stdout.readline()
        oracle_answer = str(oracle_answer, 'utf-8')

        scores = oracle_answer.split()
        if len(scores) == 8:
            scores = scores[1:]

        if self.show_scores:
            for m in range(7):
                print(f'{m+1}: {scores[m]}')
        return max(range(N_COL), key = lambda i: int(scores[i]))

class TensorFlowProabilitiesPlayer(Player):
    """
    Player that use a TensorFlow/Keras Neural Network to get the proabilitity (or score) of each
    possible moves in every situoation.

    The model must have an input layer with the same shape of the one defined in
    :func:`~GameEngine.Board.as_numpy`; the output layer must have `N_COL` nodes.
    The node with the highest probability (score) is choosen as the move. 


    Attributes
    ----------
    name: `string`, default: 'TensorFlowProabilities'
        Name used to refer to the player.
    model_path: `string`
        Path to the TensorFlow model that must be loaded.
    """
    def __init__(self, model_path, name = 'TensorFlowProabilities Player'):
        super().__init__(name)
        self.network = tf.keras.models.load_model(model_path)

    def move(self, board, moves, self_player):
        """
        See :func:`GameEngine.Player.move` for arguments explenation.
        """
        np_board = board.as_numpy(self_player)
        np_board = np_board[np.newaxis, ...]
        scores = (self.network.predict(np_board)).reshape((N_COL,))
        for c in range(N_COL):
            if len(board.board[c]) >= N_ROW:
                scores[c] = -float('inf')

        return np.argmax(scores)


class TensorFlowScorePlayer(Player):
    """
    Player that use a TensorFlow/Keras Neural Network to get the score (or score) of the
    input configuration.

    The model must have an input layer with the same shape of the one defined in
    :func:`~GameEngine.Board.as_numpy`; the output layer must have one single node.
    The output is the score of the configuartion.
     
    The player calculate the score of each allowed move he can do and then choose the best.

    Attributes
    ----------
    name: `string`, default: 'TensorFlowProabilities'
        Name used to refer to the player.
    model_path: `string`
        Path to the TensorFlow model that must be loaded.
    show_scores: `bool`, default: `False`
        If it is `True` the scores  of each move are printed on `stdout` when
        method :func:`~Players.TensorFlowScorePlayer.move` is called.
    """
    def __init__(self, model_path, show_scores=False, name = 'NeuralNetwork Player'):
        super().__init__(name)
        self.model = tf.keras.models.load_model(model_path)
        self.show_scores = show_scores

    def _valid_moves(self, board):
        return [m for m in range(N_COL) if len(board.board[m]) < N_ROW]

    def move(self, board, moves, self_player):
        """
        See :func:`GameEngine.Player.move` for arguments explenation.
        """
        valid_moves = self._valid_moves(board)
        moved_boards = []
        for m in valid_moves:
            board_copy = deepcopy(board)
            board_copy.add_move(m, self_player)
            moved_boards.append(board_copy.as_numpy(self_player))
        # Compute the scores in parallel, it's more than 5x faster than one by one calculation
        valid_moves_scores = self.model.predict(np.array(moved_boards))
        if self.show_scores:
            for c, s in zip(valid_moves, valid_moves_scores):
                print(f'{c}: {s}')
        return valid_moves[np.argmax(valid_moves_scores)]

class TwoStagePlayer(Player):
    """
    Combine two different Players.

    Opene player plays the first moves, while the closer plays the last ones.

    Attributes
    ----------
    opener: `Player`
        The player who is going to move during the opening phase.
    closer: `Player`
        The player who is going to move during the closing phase.
    open_stage: `unsigned int`
        How many moves the opening phase lasts.
    """
    def __init__(self, opener, closer, open_stage):
        super().__init__(f'{opener.name} & {closer.name}')
        self.played_moves = 0
        self.opener = opener
        self.closer = closer
        self.open_stage = open_stage
    
    def move(self, board, moves, self_player):
        """
        See :func:`GameEngine.Player.move` for arguments explenation.
        """
        self.played_moves += 1
        if self.played_moves <= self.open_stage:
            return self.opener.move(board, moves, self_player)
        else:
            return self.closer.move(board, moves, self_player)
    
    def reset(self):
        self.played_moves = 0
        self.opener.reset()
        self.closer.reset()

class PoolPlayer(Player):
    """
    Combine two different Players. 

    Use majority voting for combine the opion of different players.

    Attributes
    ----------
    players: `list(Player)`
        The players.
    n_jobs: `unsigned int`, default: `None`
        Numebr of parallel jobs that can be executed.
        Default is `None` that means max available.

    Note
    ----
    The tie-breaking startegy used it's the same as `scipy.stats.mode`.
    """
    def __init__(self, players, name = 'Pool Player', n_jobs = None):
        super().__init__(name)
        self.players = players
        self.n_jobs = n_jobs

    def _p2m(self, p, board, moves, self_player):
        return p.move(board, moves, self_player)
    
    def move(self, board, moves, self_player):
        """
        See :func:`GameEngine.Player.move` for arguments explenation.
        """
        pool = mp.pool.ThreadPool(self.n_jobs)
        async_played_moves = [pool.apply_async(self._p2m, (p, board, moves, self_player)) for p in self.players]
        played_moves = []
        for apm in async_played_moves:
            played_moves.append(apm.get())
        return int(mode(played_moves)[0])
    
    def reset(self):
        """
        See :func:`GameEngine.Player.reset` for arguments explenation.
        """
        for p in self.players:
            p.reset()


