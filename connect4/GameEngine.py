"""
The game engine for Connect4.
"""

import emoji
import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod

from .costants import N_ROW, N_COL, P1, P2, NOBODY, RED

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
    elif player == P2:
        return P1
    else:
        raise ValueError(f'Unknown player {player}')


class Player(ABC):  # pragma: no cover
    """
    Abstract class for players.

    This class has to be used as father class for Player implementations.
    All the methods an attributes that contains are need from class `Game`.

    Attributes
    ----------
    name: `string`, default: 'Player'
        Name used to refer to the player.
    """
    @abstractmethod
    def __init__(self, name = 'Player'):
        self.name = name

    @abstractmethod
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
        `int` (move)
            The move that the player would like to do.
            A move is a number in interval ``[0,N_COL)``, indicating the
            chosen column from left to right.
        Note
        ----
        The implementation must take care if the move is allowed or not.
        """
        return len(moves) % 7  # It's not garantee that the move is allowed!

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
        `True` if every call to :func:`Board.add_move` or :func:`Board.add_moves` has been processed correclty.
        `False` if a call to one of the them can be excuted beacause the column is already full.

        Note
        ----
        This does not check if the board is in a valid game situation!
        There could be a column full of coins of the same player and the valid would be still `True`!
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
        """
        Add a list of moves to the board.

        This method adds to the board a list of moves, alternating the player.
        The player that has done the first move in the list is defined by `starting`.

        Parameters
        ----------
        moves: `list(move)`
            List of moves to be added.
            A move is a number in interval ``[0,N_COL)``, indicating the
            chosen column from left to right.
        starting: `player_tag`
            Player that moves first.
        """
        player = starting
        for m in moves:
            self.add_move(m, player)
            player = change_player(player)

    def add_move(self, move, player):
        """
        Add a move to the board, if it is possible.

        Parameters
        ----------
        player: `player_tag`
            The player to be changed.
        Returns
        -------
        `player_tag`
            The opposite player from the one given.
        """
        if not self.is_allowed(move):
            self.valid = False
        self.board[move].append(player)

    def is_allowed(self, move):
        """
        Check if applying `move` is allowed in the current board.

        Parameters
        ----------
        move: `int(move)`
            The move to be checked.
        Returns
        -------
        `bool`
            `True` if the move is allowed, `False` instead.
        """
        if len(self.board[move]) >= N_ROW or not self.valid:
            return False
        return True

    def evalutate(self, winner_points = True):
        """
        Check if one of the player .

        The function can also compute the number of cell that contributes to the victory.
        A cell is contributing to the victory iff it's filled by a winner coin,
        and there is a line (vertical, horizontal or diagonal) of at least 4 consecutive
        contributing cell that contains it. This number it's named *winner points*.

        Parameters
        ----------
        winner_points: `bool`
            If `True` calculate number of cells contributing to the victory.
            If `False` calculate only the winner and set the *winner points* to 0.

            If `False` the function has better performance, so you should calculate the winning
            points only if you really need.
        Returns
        -------
        `tuple(player_tag, unsigned int)`
            The first element of the tuple is the winner, if any. Otherwise is `NOBODY`.
            The second element are the *winner points*. If `winner_points = False` or there's
            no winner, it'a 0 by default.

            If the board is not valid, the reurn is `tuple(None, None)`
        Note
        ----
        The method assumes that the board is in avalid game situation.
        If not, it's an undefined behavior.
        """
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
                    for k in range(1, 4):
                        row_win = (self.board[j + k][i] == this_cell)
                        if not row_win:
                            break
                    if row_win:
                        winner = this_cell
                        if not winner_points:
                            break
                        winning_cells.add((j, i))
                        for k in range(1, 4):
                            winning_cells.add((j + k, i))
                except IndexError:
                    pass
                # Cols
                try:
                    col_win = True
                    for k in range(1, 4):
                        col_win = (self.board[j][i + k] == this_cell)
                        if not col_win:
                            break
                    if col_win:
                        winner = this_cell
                        if not winner_points:
                            break
                        winning_cells.add((j, i))
                        for k in range(1, 4):
                            winning_cells.add((j, i + k))
                except IndexError:
                    pass

                # Diag direct
                try:
                    diag_d_win = True
                    for k in range(1, 4):
                        diag_d_win = (self.board[j + k][i + k] == this_cell)
                        if not diag_d_win:
                            break
                    if diag_d_win:
                        winner = this_cell
                        if not winner_points:
                            break
                        winning_cells.add((j, i))
                        for k in range(1, 4):
                            winning_cells.add((j + k, i + k))
                except IndexError:
                    pass

                # Diag inverse
                try:
                    diag_i_win = True
                    for k in range(1, 4):
                        diag_i_win = (self.board[j + k][i - k] == this_cell) and i - k >= 0  # if i-k==-1 we are checking the top element of the column
                        if not diag_i_win:
                            break
                    if diag_i_win:
                        winner = this_cell
                        if not winner_points:
                            break
                        winning_cells.add((j, i))
                        for k in range(1, 4):
                            winning_cells.add((j + k, i - k))
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

    def __str__(self):  # pragma: no cover
        board_str = ''
        for i in range(N_ROW):
            for j in range(N_COL):
                board_str += '|'
                try:
                    if self.board[j][N_ROW - 1 - i] == RED:
                        board_str += ':red_circle:'
                    else:
                        board_str += ':yellow_circle:'
                except IndexError:
                    board_str += ':black_large_square:'
            board_str += '|\n'
        # board_str += '-'*(2*N_COL+1) + '\n'
        for i in range(1, N_COL + 1):
            board_str += f' :keycap_{i}: '

        return emoji.emojize(board_str)


class Game():
    """

    Player1 is using red color, player2 yellow.
    Columns are numbered from 0 to `N_COL`, left to right.

    Attributes
    ----------
    starter: `player_tag`
        The player who starts the game.

    turn: `player_tag`
        The player that has to move next.

    board: `Board`
        Board object of the game.

    finish: `bool`
        `True` only if the game is finished.

    moves: `list(move)`
        List of the moves played during the game.
        It's assumed that player are alternating, starting from `self.starter`.

    winner: `player_tag`
        `NOBODY` if the game is not finished or ended with a tie.
        Otherwise it's the tag of the winner.

    winner_points: `int`
        Points scored by the winner. If `winner=NOBODY` then `winner_point=0`.


    Parameters
    ----------
    player1: Player
        Player 1 object.

    player2: Player
        Player 2 object.

    starter: `player_tag`, default: `P1`
        The player that is sterting the game.
    """

    def __init__(self, player1, player2, starter = P1):
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
        """
        Check if the game is finished.

        Returns
        -------
        `bool`
            `True` if the game is ended, otherwise it returns `False`.
        """
        self.winner, self.winner_points = self.board.evalutate()
        self.finish = (self.winner != NOBODY) or (len(self.moves) == N_ROW * N_COL)
        return self.finish

    def insert_coin(self, move, player):
        """
        Player make a move.

        Parameters
        ----------
        move: `move`
            Move that has to be played.
        player: `player_tag`
            Player that make the move.
            Raise `AssertionError` if `player != self.turn`.

        Note
        ----
        This method is meant for internal use only. If you are going to use it, make sure that you know
        what you are doing.
        """
        assert(self.turn == player)
        if not self.board.is_allowed(move):
            raise ValueError(f'The column {move} is already full!')
        self.board.add_move(move, player)
        self.moves.append(move)
        if self.check_finish():
            self.p1.reset()
            self.p2.reset()

    def next(self):
        """
        Make the current turn player move.

        This function calls the method `move` of the Player objects.
        """
        if self.finish:
            return

        if self.turn == P1:
            self.insert_coin(self.p1.move(self.board, self.moves, P1), P1)
        else:
            self.insert_coin(self.p2.move(self.board, self.moves, P2), P2)
        self.turn = change_player(self.turn)

    def play_all(self):
        """
        Play the game untill the finish.
        """
        while not self.finish:
            self.next()

    def __str__(self):  # pragma: no cover
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
                message_str += "It's a tie!"

        return f'{self.board}\n {message_str}'


class Tournament():
    """
    Play a tournament between two players.

    Multiple games are played between two player and statics is collected to compare them.

    Attributes
    ----------
    player1: Player
        Player 1 object.

    player2: Player
        Player 2 object.

    starting_criteria: `string`
        Could have 4 values:
        - `'alternate'`: alternate the staring player, starting from `P1`.
        - `'random'`: choose at random the player that will start the game.
        - `'p1'`: Player 1 is always staring.
        - `'p2'`: Player 2 is always staring.

    last_starter: `player_tag`
        The player who has started the last game. At the beginnign is `NOBODY`.

    counter: `np.array(shape=(3,3), dtype=np.uint32)`
        The object who is collecting the statics of the tournament.

        Examples
        --------
        - `counter[P1][P1]` -> Games started by `P1` winned by `P1`.
        - `counter[P2][NOBODY]` -> Games started by `P2` winned by `NOBODY`.
    """

    def next_starter(self):
        """
        Choose the player who is staring next game.

        It also update the `self.last_starter` attribute.

        Returns
        -------
        `player_tag`
            The next starter.
        """
        if self.starting_criteria == 'alternate':
            if self.last_starter == NOBODY:
                self.last_starter = P1
            else:
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
        """
        Reset all the statics collected by the counter.
        """
        self.counter = np.zeros((3, 3), dtype=np.uint32)

    def __init__(self, player1, player2, starting_criteria = 'alternate'):
        self.player1 = player1
        self.player2 = player2
        self.reset_counter()
        self.starting_criteria = starting_criteria
        self.last_starter = NOBODY

    def play_games(self, to_be_played = 1):
        """
        Play a fixed number of games.

        Parameters
        ----------
        to_be_played: `unsigned int`
            Number of matches to be played.
        """
        for _ in tqdm(range(to_be_played)):
            g = Game(self.player1, self.player2, self.next_starter())
            g.play_all()
            self.counter[self.last_starter][g.winner] += 1

    def finished(self):
        """
        Methods that must be called when the Tournament is ended.

        It calls the `finished` method on the Players object.
        """
        self.player1.finished()
        self.player2.finished()

    def __str__(self):  # pragma: no cover
        return f"""Started by {self.player1.name}, played {self.counter[P1][NOBODY]+self.counter[P1][P1]+self.counter[P1][P2]}:
    Tied {self.counter[P1][NOBODY]} games
    {self.player1.name} won {self.counter[P1][P1]} games
    {self.player2.name} won {self.counter[P1][P2]} games
Started by {self.player2.name}, played {self.counter[P2][NOBODY]+self.counter[P2][P1]+self.counter[P2][P2]}:
    Tied {self.counter[P2][NOBODY]} game
    {self.player1.name} won {self.counter[P2][P1]}
    {self.player2.name} won {self.counter[P2][P2]}"""
