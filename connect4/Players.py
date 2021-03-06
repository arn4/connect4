"""
Some players implementations for connect4.
"""

import random as rnd
import numpy as np
import subprocess

from .GameEngine import *
from .algorithms import minimax, alphabeta, WORSE_SCORE, MOVES_ORDER, MIN_SCORE_STEP

class ConsolePlayer(Player):
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

    def move(self, board, moves, self_player):
        """
        stdin input for next move.

        The move is to be inserted in the console is a number from 1 ro N_COL,
        from left to right.
        Argouments are unused. See :func:`GameEngine.Player.move` for arguments explenation.
        """
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
            c = rnd.randint(0,6)
            if len(board[c]) < 6:
                return c

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
        return [minimax(moves + [m], self.depth-1, change_player(self_player)) for m in range(N_COL)]

    def move(self, board, moves, self_player):
        """
        See :func:`GameEngine.Player.move` for arguments explenation.
        """
        scores = self.get_opponent_scores_given_move(moves, self_player)
        for m in range(N_COL):
            try:
                scores[m] *= -1
            except TypeError:
                scores[m] = -float('inf')
        if self.show_scores:
            for m in range(N_COL):
                print(f'{m+1}: {scores[m]:.1f}')

        return max(range(N_COL), key = lambda i: scores[i] if scores[i] is not None else 2*WORSE_SCORE)

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
                            depth = self.depth-1,
                            self_player = change_player(self_player))
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
        for i in range(len(MOVES_ORDER)):
            try:
                scores[MOVES_ORDER[i]] += np.random.normal(0., self.noise)
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

    def game_finished(self):
        """
        Kill `oracle` process.

        See :func:`GameEngine.Player.game_finished`
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
