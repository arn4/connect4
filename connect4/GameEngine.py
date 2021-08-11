"""
The game engine for Connect4.
"""

import emoji
import numpy as np
from tqdm import tqdm

from .costants import *

##########
# PLAYER #
##########

def change_player(player):
    """
    Return the opposite player.

    Parameters
    ----------
    player: `player_tag`
        The player to be changed.
    Returns
    -------
    `player_tag`
        The opposite player from the one given.
    """
    if player == P1:
        return P2
    else:
        return P1

class Player():
    """
    Base class for players.

    This class has to be used as father class for Player implementations.
    All the methods an attributes that contains are need from class `Game`.

    Attributes
    ----------
    name: `string`, default: 'Player'
        Name used to refer to the player.
    """
    def __init__(self, name = 'Player'):
        self.name = name

    def move(self, board, moves, self_player):
        """
        Make a move.

        This method should return which move the player would like to do
        given the game situation.

        Parameters
        ----------
        board: `Board`
            Board object of the actual situation.
        moves: `list`
            List of moves that  have been done throught the game.
            A move is a number in interval ``[0,N_COL)``, indicating the
            chosen column from left to right.
            Moves do not distinguish wich player did it.
        self_player: `player_tag`
            Tag of the current player.
        Returns
        -------
        `int`
            The move that the player would like to do.
            A move is a number in interval ``[0,N_COL)``, indicating the
            chosen column from left to right.
        Note
        ----
        The implementation must take care if the move is allowed or not.
        """
        return len(moves) % 7 # It's not garantee that the move is allowed!

    def finished(self):
        """
        Method to be called when the player is not playing anymore.
        """
        pass

    def reset(self):
        """
        Reset the internal state of a player and get ready for another game.
        """
        pass

    def probabilities(self, board, moves, self_player):
        """
        Return the vector of the probability for each move.

        These is not need for playing games, but only for supervising them.
        Returns
        -------
        `numpy.array(shape=(N_COL,))`
            For each column gives the probability that a player 
        """
        raise NotImplementedError("This player has not method probabilities()")

#########
# BOARD #
#########

class Board():
    """
    Board of a Game. Stores the position and thes history of the moves
    of two player (`P1` and `P2`).

    Attributes
    ----------
    valid: `bool`
        Is the board a valid game situation.
    board: `list(list(player_tag))`
        For each column stores the inserted coins from bottom to top.
        All list are 0-indexed.

        You shouldn't modify this attribute directly. Use :func:`Board.add_move`
        or :func:`Board.add_moves` instead.

        Examples
        --------
        `board[0][5] == P1`:
            means that in the first column from left, third column
            from bottom there is a coin from Player 1.
        `board[3][2] == P1`:
            means that in the fourth column from left, sixth column
            from bottom there is a coin from Player 2.
    """
    def __init__(self, moves = None, starter = None):
        self.empty()
        self.valid = True
        if moves is not None:
            if starter is None:
                raise ValueError('starter cannot be None when moves is not None.')
            self.add_moves(moves, starter)

    def empty(self):
        """
        Empty the board.
        """
        self.board = [[] for _ in range(N_COL)]

    def add_moves(self, moves, starting):
        player = starting
        for m in moves:
            self.add_move(m, player)
            player = change_player(player)

    def add_move(self, move, player):
        if not self.is_allowed(move):
            self.valid = False
        self.board[move].append(player)


    def is_allowed(self, move):
        if len(self.board[move]) >= N_ROW or not self.valid:
            return False
        return True

    def evalutate(self, winner_points = True):
        if not self.valid:
            return None, None
        winning_cells = set()
        winner = NOBODY
        for i in range(N_ROW):
            if winner != NOBODY and not winner_points:
                break
            for j in range(N_COL):
                try:
                    this_cell = self.board[j][i]
                except IndexError:
                    continue
                # Rows
                try:
                    row_win = True
                    for k in range(1,4):
                        row_win = (self.board[j+k][i] == this_cell)
                        if not row_win:
                            break
                    if row_win:
                        winner = this_cell
                        if not winner_points:
                            break
                        winning_cells.add((j,i))
                        for k in range(1,4):
                            winning_cells.add((j+k,i))
                except IndexError:
                    pass
                # Cols
                try:
                    col_win = True
                    for k in range(1,4):
                        col_win = (self.board[j][i+k] == this_cell)
                        if not col_win:
                            break
                    if col_win:
                        winner = this_cell
                        if not winner_points:
                            break
                        winning_cells.add((j,i))
                        for k in range(1,4):
                            winning_cells.add((j,i+k))
                except IndexError:
                    pass

                # Diag direct
                try:
                    diag_d_win = True
                    for k in range(1,4):
                        diag_d_win = (self.board[j+k][i+k] == this_cell)
                        if not diag_d_win:
                            break
                    if diag_d_win:
                        winner = this_cell
                        if not winner_points:
                            break
                        winning_cells.add((j,i))
                        for k in range(1,4):
                            winning_cells.add((j+k,i+k))
                except IndexError:
                    pass

                # Diag inverse
                try:
                    diag_i_win = True
                    for k in range(1,4):
                        diag_i_win = (self.board[j+k][i-k] == this_cell) and i-k>=0 # if i-k==-1 we are checking the top element of the column
                        if not diag_i_win:
                            break
                    if diag_i_win:
                        winner = this_cell
                        if not winner_points:
                            break
                        winning_cells.add((j,i))
                        for k in range(1,4):
                            winning_cells.add((j+k,i-k))
                except IndexError:
                    pass

        if winner_points:
            return winner, len(winning_cells)
        else:
            return winner, 0

    def as_numpy(self, one_player):
        """
        Return the board as numpy.array.

        The returned object has shape = (N_ROW, N_COL) and dtype = np.int8.
        There's a 1 where `one_player` has a coin, there's -1 where the other
        player has a coin. Empty cell are 0s.

        Parameters
        ----------
        one_player: `player_tag`
            The tag of the player represented with 1s in the numpy board.
        
        Returns
        -------
        `np.array`
            The numpy board.
        """
        np_board = np.zeros((N_ROW, N_COL), dtype=np.int8)
        for icol, col in enumerate(self.board):
            for irow, cell in enumerate(col):
                if cell == one_player:
                    np_board[irow][icol] = int(1)
                else:
                    np_board[irow][icol] = int(-1)
        return np_board

    def __str__(self):
        board_str = ''
        for i in range(N_ROW):
            for j in range(N_COL):
                board_str += '|'
                try:
                    if self.board[j][N_ROW-1-i] == RED:
                        board_str += ':red_circle:'
                    else:
                        board_str += ':yellow_circle:'
                except IndexError:
                    board_str += ':black_large_square:'
            board_str += '|\n'
        #board_str += '-'*(2*N_COL+1) + '\n'
        for i in range(1,N_COL+1):
            board_str += f' :keycap_{i}: '

        return emoji.emojize(board_str)



class Game():
    """
    Player 1 plays with RED, player 2 with YELLOW.
    Player 1 always starts.

    Columns are numbered from 0 to 6, left to right.
    """

    def  __init__(self, player1, player2, starter = P1):
        self.starter = starter
        self.turn = starter
        self.p1 = player1
        self.p2 = player2
        self.board = Board()
        self.finish = False
        self.moves = list()

        self.winner = NOBODY
        self.winner_points = 0


    def check_finish(self):
        self.winner, self.winner_points = self.board.evalutate()
        self.finish = (self.winner != NOBODY) or (len(self.moves) == N_ROW*N_COL)
        return self.finish

    def insert_coin(self, move, player):
        if not self.board.is_allowed(move):
            raise ValueError(f'The column {move} is already full!')
        self.board.add_move(move, player)
        self.moves.append(move)
        if self.check_finish():
            self.p1.reset()
            self.p2.reset()

    def next(self):
        if self.finish:
            return

        if self.turn == P1:
            self.insert_coin(self.p1.move(self.board, self.moves, P1), P1)
        else:
            self.insert_coin(self.p2.move(self.board, self.moves, P2), P2)
        self.turn = change_player(self.turn)

    def play_all(self):
        while not self.finish:
            self.next()

    def __str__(self):
        message_str = ''
        if not self.finish:
            message_str += 'Next moves is up to '
            if self.turn == P1:
                message_str += self.p1.name
            else:
                message_str += self.p2.name
            message_str += '\n'
        else:
            if self.winner != NOBODY:
                if self.winner == P1:
                    winner_name = self.p1.name
                else:
                    winner_name = self.p2.name
                message_str += f'Winner is {winner_name}, with {self.winner_points} points'
            else:
                message_str += f"It's a tie!"

        return f'{self.board}\n {message_str}'


class Tournament():
    def next_starter(self):
        if self.starting_criteria == 'alternate':
            self.last_starter = change_player(self.last_starter)
        elif self.starting_criteria == 'random':
            self.last_starter = np.random.choice([P1, P2])
        elif self.starting_criteria == 'p1':
            self.last_starter = P1
        elif self.starting_criteria == 'p2':
            self.last_starter = P2
        else:
            raise ValueError(f'Unknwon starting_criteria = {self.starting_criteria}')
        return self.last_starter

    def reset_counter(self):
        self.counter = np.zeros((3,3), dtype=np.uint32)
        
    def __init__(self, player1, player2, starting_criteria = 'alternate'):
        self.player1 = player1
        self.player2 = player2
        self.reset_counter()
        self.starting_criteria = starting_criteria
        self.last_starter = NOBODY

    def play_games(self, to_be_played = 1):
        for _ in tqdm(range(to_be_played)):
            g = Game(self.player1, self.player2, self.next_starter())
            g.play_all()
            # if (g.winner == P1):
            #     print(g)
            self.counter[self.last_starter][g.winner] += 1
    def finished(self):
        self.player1.finished()
        self.player2.finished()

    def __str__(self):
        return f"""Started by {self.player1.name}, played {self.counter[P1][NOBODY]+self.counter[P1][P1]+self.counter[P1][P2]}:
    Tied {self.counter[P1][NOBODY]} games
    {self.player1.name} won {self.counter[P1][P1]} games
    {self.player2.name} won {self.counter[P1][P2]} games
Started by {self.player2.name}, played {self.counter[P2][NOBODY]+self.counter[P2][P1]+self.counter[P2][P2]}:
    Tied {self.counter[P2][NOBODY]} game
    {self.player1.name} won {self.counter[P2][P1]}
    {self.player2.name} won {self.counter[P2][P2]}"""
            