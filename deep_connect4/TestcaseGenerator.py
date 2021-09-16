import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

from .GameEngine import Game
from .costants import P1


class SupervisedGame(Game):
    """
    Play a game between two players, with a supervisor that annotates which moves he would do in each situation.
    """

    def __init__(self, player1, player2, supervisor, starter = P1):
        super().__init__(player1, player2, starter)
        self.supervisor = supervisor
        self.testcases = []

    def next(self):
        """
        Next move in the game.

        Record what supervisor would do in this situation.
        """
        self.testcases.append((self.board.as_numpy(self.turn), self.supervisor.probabilities(self.board, self.moves, self.turn)))
        Game.next(self)

    def get_testcases(self, transient = 0):
        """
        Return the testcases generated during the game.

        This method should return which move the player would like to do
        given the game situation.

        Parameters
        ----------
        transient: `int`:
            Number of initial moves to not be considered when genereting testcases.

        Returns
        -------
        `list(tuple(numpy.array(shape(N_ROW, NCOLS)), numpy.array(shape(NCOLS))))`
            List of tuples. Each tuple contains the board, in numpy.array format :func:Board.as_numpy
        """
        return self.testcases[transient:]


class DatasetGenerator():
    def _run_game(self):
        p1 = self.player1[0](*self.player1[1])
        p2 = self.player2[0](*self.player2[1])
        sv = self.supervisor[0](*self.supervisor[1])
        sg = SupervisedGame(p1, p2, sv)
        sg.play_all()
        p1.finished()
        p2.finished()
        return sg.get_testcases()

    def __init__(self, player1, player2, supervisor):
        self.player1 = player1
        self.player2 = player2
        self.supervisor = supervisor

    def run(self, n_games, n_jobs = 1):
        pool = Pool(processes=n_jobs)
        async_games = [pool.apply_async(self._run_game, ()) for _ in range(n_games)]
        games = []
        for ag in tqdm(async_games):
            games += ag.get()
        pool.terminate()
        testcases_board, testcases_move = zip(*games)
        self.boards = np.array(testcases_board)
        self.moves = np.array(testcases_move)

    def save(self, filename):
        try:
            np.savez_compressed(filename, boards=self.boards, moves=self.moves)
        except BaseException:
            raise RuntimeError('Unable to save Dataset. Check you have called run()')
