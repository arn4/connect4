
import emoji

from costants import *

def change_player(player):
    if player == P1:
        return P2
    else:
        return P1

class Player():
    def __init__(self, name = 'Player'):
        self.name = name

    def move(self, board, moves):
        return len(moves) % 7 # It's not garantee that the move is allowed!

    def game_finished(self):
        pass

class Board():
    def __init__(self, moves = None):
        self.empty()
        self.valid = True
        if moves is not None:
            self.add_moves(moves, P1)

    def empty(self):
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

    Columns are from 0 to 6.
    """

    def  __init__(self, player1, player2, starter = P1):
        self.turn = P1
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
            self.p1.game_finished()
            self.p2.game_finished()

    def next(self):
        if self.finish:
            return False

        if self.turn == P1:
            self.insert_coin(self.p1.move(self.board, self.moves, P1), P1)
        else:
            self.insert_coin(self.p2.move(self.board, self.moves, P2), P2)
        self.turn = change_player(self.turn)

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
                message_str += f'Winner is {self.winner}, with {self.winner_points} points'
            else:
                message_str += f"It's a tie!"

        return f'{self.board}\n {message_str}'
