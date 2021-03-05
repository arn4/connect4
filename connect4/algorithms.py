
from .GameEngine import *

## Minimax algorithm parameters
POINT_TO_SCORE = 0
"""
Factor of conversion between point of winner (see `GameEngine`) and score for the
algorithms.
"""
WORSE_SCORE = -10000
"""
Worse score. It's used as indicator of an orrible move.
"""

# Heurist order for improve alpha-beta pruning
MOVES_ORDER = []
"""
Moves ordered by heuristic.
"""
if N_COL % 2 == 1:
    half = int(N_COL/2)
    MOVES_ORDER.append(half)
    for i in range(1,half+1):
        MOVES_ORDER.append(half-i)
        MOVES_ORDER.append(half+i)
else:
    half = int(N_COL/2)
    for i in range(half):
        MOVES_ORDER.append(half-1-i)
        MOVES_ORDER.append(half+i)

MIN_SCORE_STEP = 0.5
"""
The minimum possible difference between two different scores.
"""


def minimax(moves, depth, self_player):
    """
    This is a NegaMax implemetation with fixed depth.

    Compute the best score reachable from the actual situation.

    Parameters
    ----------
    moves: `list`
        List of moves that  have been done throught the game.
        A move is a number in interval ``[0,N_COL)``, indicating the
        chosen column from left to right.
        Moves do not distinguish wich player did it.
    depth: `int`
        Depth of the tree to be explored.
    self_player: `player_tag`
        Tag of the current player.

    Returns
    -------
    float:
        Best scores reachable from the actual situation.
    """
    board = Board(moves)
    if not board.valid:
        return None
    w, wp = board.evalutate()

    if w != NOBODY:
        score = (N_ROW*N_COL - len(moves))/2 + POINT_TO_SCORE*max(0,wp-4)
        if w == self_player:
            return score
        else:
            return -score
    if depth == 0:
        return 0

    best_score = WORSE_SCORE
    for m in range(N_COL):
        try:
            best_score = max(best_score, -minimax(moves+[m],depth-1, change_player(self_player)))
        except TypeError:
            pass
    return best_score

def alphabeta(moves, depth, alpha, beta, self_player):
    """
    This is a NegaMax implemetation with fixed depth and alpha-beta pruning.

    Compute the best score reachable from the actual situation.
    We are assuming that the score of a node is always greater or equal to the score of its children.
    Parameters
    ----------
    moves: `list`
        List of moves that  have been done throught the game.
        A move is a number in interval ``[0,N_COL)``, indicating the
        chosen column from left to right.
        Moves do not distinguish wich player did it.
    depth: `int`
        Depth of the tree to be explored.
    self_player: `player_tag`
        Tag of the current player.
    alpha: float
        Alpha score of the algorithm.
    beta: float
        Beta score of the algorithm.
    Returns
    -------
    float:
        Best scores reachable from the actual situation.
    """
    board = Board(moves)
    if not board.valid:
        return None
    w, wp = board.evalutate(winner_points = False)
    score = (N_ROW*N_COL - len(moves))/2
    if w != NOBODY:
        if w == self_player:
            return score
        else:
            return -score
    if depth == 0:
        return 0

    if score <= alpha:
        return alpha

    best_score = WORSE_SCORE
    for m in MOVES_ORDER:
        try:
            best_score = max(best_score, -alphabeta(moves+[m],depth-1, -beta, -best_score, change_player(self_player)))
            if best_score >= beta:
                return beta
        except TypeError:
            pass
    return best_score
