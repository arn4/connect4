import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tqdm import tqdm
import multiprocessing as mp

from connect4.costants import *
from connect4.GameEngine import *
from connect4.Players import *

WON_GAME_SCORE = 10.
TIED_GAME_SCORE = 1.
REDUCTION = 0.9


class EpsilonGreadyPlayer(TensorFlowScorePlayer):
    def __init__(self, nn_model, epsilon):
        self.model = nn_model
        self.epsilon = epsilon

    def move(self, board, moves, self_player):
        valid_moves = self._valid_moves(board)
        if np.random.uniform() < self.epsilon:
            return np.random.choice(valid_moves)
        else:
            return super().move(board, moves, self_player)

class RLNeuralNetworkTrainer():
    def __init__(self, neural_network_model, n_players, model_name):
        self.models = [neural_network_model() for i in range(n_players)]
        self.n_players = n_players
        self.loss_object = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam()

        self.model_name = model_name

            
    # @tf.function
    def _train_step(self, model, boards_batch, scores_batch):
        with tf.GradientTape() as tape:
            batch_predictions = model(boards_batch, training=True)
            loss_on_batch = self.loss_object(scores_batch, batch_predictions)
            gradients = tape.gradient(loss_on_batch, model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    def _game_with_scores(self, player1, player2, epsilon, reduction):
        g = Game(EpsilonGreadyPlayer(self.models[player1], epsilon), EpsilonGreadyPlayer(self.models[player2], epsilon))
        collected_states = {P1:[], P2:[]}
        while not g.finish:
            collected_states[g.turn].append(g.board.as_numpy(g.turn))
            g.next()

        collected_scores = {}
        for P in [P1,P2]:
            collected_states[P].append(g.board.as_numpy(P))
            if g.winner == P:
                score = WON_GAME_SCORE
            elif g.winner == change_player(P):
                score = -WON_GAME_SCORE
            else:
                score = TIED_GAME_SCORE
            n_moves = len(collected_states[P])
            collected_scores[P] = [(reduction)**(2*(n_moves - 1 - i))*score for i in range(n_moves)]
        
        assert(len(collected_states[P1]) == len(collected_scores[P1]))
        assert(len(collected_states[P2]) == len(collected_scores[P2]))
        
        self.boards_batch[player1] += collected_states[P1]
        self.scores_batch[player1] += collected_scores[P1]
        self.boards_batch[player2] += collected_states[P2]
        self.scores_batch[player2] += collected_scores[P2]

    def train(self, episodes, epsilon_decay, batch_size, use_symmetry = True, save_every = None, n_jobs = None):
        if save_every is None:
            save_every = episodes

        epsilon = 1.
        self.boards_batch = [[] for p in range(self.n_players)]
        self.scores_batch = [[] for p in range(self.n_players)]

        for episode in tqdm(range(1, episodes + 1)):
            # Thread seems to be the best choiche in this case.
            # Probably parallelizing with sprocees is too demanding for the operating system.
            # No parallelization is instead worse
            games_pool = mp.pool.ThreadPool(n_jobs)
            # games_pool = mp.Pool(n_jobs)
            for p1, p2 in zip(range(self.n_players), range(self.n_players)):
                # self._game_with_scores(p1, p2, epsilon, REDUCTION)
                games_pool.apply_async(self._game_with_scores, (p1, p2, epsilon, REDUCTION))
            games_pool.close()
            games_pool.join()

            for p in range(self.n_players):
                if len(self.boards_batch[p]) > batch_size:
                    np_boards_batch = np.array(self.boards_batch[p])
                    np_scores_batch = np.array(self.scores_batch[p])
                    if use_symmetry:
                        np_symmetrized_boards_batch = np.flip(np_boards_batch, 2)
                        np_boards_batch = np.concatenate([np_boards_batch, np_symmetrized_boards_batch])
                        np_scores_batch = np.concatenate([np_scores_batch, np_scores_batch])
                        assert(np_boards_batch.shape[0] == np_scores_batch.shape[0])
                    self._train_step(self.models[p], np_boards_batch, np_scores_batch)
                    self.boards_batch[p] = []
                    self.scores_batch[p] = []
            
            for p in range(self.n_players):
                if episode % save_every == 0:
                    print(f'Saving {self.model_name}_ep-{episode}_id-{p}')
                    self.models[p].save(self.model_name + f'_ep-{episode}_id-{p}')

            epsilon *= epsilon_decay
