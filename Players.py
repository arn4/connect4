from random import randint
from numpy.random import normal
import subprocess


from GameEngine import *
from algorithms import minimax, alphabeta, WORSE_SCORE, MOVES_ORDER, MIN_SCORE_STEP

class ConsolePlayer(Player):
    def __init__(self, name = 'Console Player'):
        super().__init__(name)

    def move(self, board, moves, self_player):
        return int(input('Which column? '))-1

class RandomPlayer(Player):
    def __init__(self, name = 'Random Player'):
        super().__init__(name)

    def move(self, board, moves, self_player):
        while True:
            c = randint(0,6)
            if len(board[c]) < 6:
                return c

class MiniMaxPlayer(Player):
    def __init__(self, depth, show_scores = False, name = 'MinMaxPlayer'):
        super().__init__(name)
        self.depth = depth
        self.scores = show_scores

    def get_opponent_scores_given_move(self, moves, self_player):
        """
        Can also contain None object if the move is not allowed
        """
        return [minimax(moves + [m], self.depth-1, change_player(self_player)) for m in range(N_COL)]

    def move(self, board, moves, self_player):
        scores = self.get_opponent_scores_given_move(moves, self_player)
        for m in range(N_COL):
            try:
                scores[m] *= -1
            except TypeError:
                scores[m] = -float('inf')
        if self.scores:
            for m in range(N_COL):
                print(f'{m+1}: {scores[m]:.1f}')

        return max(range(N_COL), key = lambda i: scores[i] if scores[i] is not None else 2*WORSE_SCORE)

class AlphaBetaPlayer(MiniMaxPlayer):
    def __init__(self, depth, show_scores = False, name = 'AlphaBetaPlayer'):
        super().__init__(depth, show_scores, name)

    def get_opponent_scores_given_move(self, moves, self_player):
        return [alphabeta(moves = moves + [m],
                            alpha = WORSE_SCORE,
                            beta = -WORSE_SCORE,
                            depth = self.depth-1,
                            self_player = change_player(self_player))
                for m in range(N_COL)]

class CenteredAlphaBetaPlayer(AlphaBetaPlayer):
    """
    We use the following heuristic:
        If 2 moves has the same score the one nearer to the center in chosen.
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
    Overlap a Gaussian Noise to AlphaBeta score to not be deterministic.
    """
    def __init__(self, depth, noise, show_scores = False, name = 'NoisyAlphaBetaPlayer'):
        super().__init__(depth, show_scores, name)
        self.noise = noise

    def get_opponent_scores_given_move(self, moves, self_player):
        scores = super().get_opponent_scores_given_move(moves, self_player)
        for i in range(len(MOVES_ORDER)):
            try:
                scores[MOVES_ORDER[i]] += normal(0., self.noise)
            except TypeError:
                pass
        return scores

class PerfectPlayer(Player):
    def __init__(self, executable_path, show_scores = False, name = 'PerfectPlayer'):
        super().__init__(name)

        self.oracle = subprocess.Popen([executable_path, '-a'],
                                        stdout = subprocess.PIPE,
                                        stdin = subprocess.PIPE,
                                        stderr = subprocess.PIPE)
        self.oracle.stderr.readline()
        self.scores = show_scores

    def game_finished(self):
        self.oracle.terminate()

    def move(self, board, moves, self_player):
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

        if self.scores:
            for m in range(7):
                print(f'{m+1}: {scores[m]}')
        return max(range(N_COL), key = lambda i: int(scores[i]))
