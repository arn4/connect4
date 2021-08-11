import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tqdm import tqdm

from connect4.costants import *
from connect4.GameEngine import *
from connect4.Players import *

WON_GAME_SCORE = 10.
REDUCTION = 0.9


class EpsilonGreadyPlayer(NeuralNetwrokScorePlayer):
    def __init__(self, nn_model, epsilon):
        self.model = nn_model
        self.epsilon = epsilon

    def move(self, board, moves, self_player):
        valid_moves = self._valid_moves(board)
        if np.random.uniform() < self.epsilon:
            return np.random.choice(valid_moves)
        else:
            return super().move(board, moves, self_player)

class RLNeuralNetwork(Model):
    def __init__(self):
        super(RLNeuralNetwork, self).__init__()
        self.flatten = Flatten(input_shape=(N_ROW, N_COL))
        self.hidden1 = Dense(100, activation='relu')
        self.hidden2 = Dense(70, activation='relu')
        self.hidden3 = Dense(50, activation='relu')
        self.hidden4 = Dense(50, activation='relu')
        self.outlayer = Dense(1)

    def call(self, x):
        x = self.flatten(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        return self.outlayer(x)

class RLNeuralNetworkTrainer():
    def __init__(self, neural_network_model, model_name):
        self.model = neural_network_model()
        self.loss_object = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss = tf.keras.metrics.Mean(name='train_loss')

        self.model_name = model_name

            
    @tf.function
    def _train_step(self, boards_batch, scores_batch):
        with tf.GradientTape() as tape:
            batch_predictions = self.model(boards_batch, training=True)
            loss_on_batch = self.loss_object(scores_batch, batch_predictions)
        gradients = tape.gradient(loss_on_batch, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.loss(loss_on_batch)

    def _game_with_scores(self, epsilon, reduction):
        g = Game(EpsilonGreadyPlayer(self.model, epsilon), EpsilonGreadyPlayer(self.model, epsilon))
        collected_states = []
        while not g.finish:
            collected_states.append(g.board.as_numpy(g.turn))
            g.next()
        collected_states.append(g.board.as_numpy(change_player(g.winner)))
        collected_states.append(g.board.as_numpy(g.winner))

        n_moves = len(collected_states)
        collected_scores = [(-reduction)**(n_moves - 1 - i)*WON_GAME_SCORE for i in range(n_moves)]
        
        assert(len(collected_states) == len(collected_scores))
        return collected_states, collected_scores



    def train(self, episodes, epsilon_decay, batch_size, use_symmetry = True, save_every = None):
        if save_every is None:
            save_every = episodes

        epsilon = 1.
        boards_batch = []
        scores_batch = []

        for episode in tqdm(range(1, episodes + 1)):
            # Reset the metrics at the start of the next epoch
            self.loss.reset_states()

            game_states, game_scores = self._game_with_scores(epsilon, REDUCTION)
            boards_batch += game_states
            scores_batch += game_scores

            if len(boards_batch) > batch_size:
                np_boards_batch = np.array(boards_batch)
                np_scores_batch = np.array(scores_batch)
                if use_symmetry:
                    np_symmetrized_boards_batch = np.flip(np_boards_batch, 2)
                    np_boards_batch = np.concatenate([np_boards_batch, np_symmetrized_boards_batch])
                    np_scores_batch = np.concatenate([np_scores_batch, np_scores_batch])
                    assert(np_boards_batch.shape[0] == np_scores_batch.shape[0])
                self._train_step(np_boards_batch, np_scores_batch)
                boards_batch = []
                scores_batch = []
            
            if episode % save_every == 0:
                print(f'Saving {self.model_name}_ep-{episode}')
                self.model.save(self.model_name + f'_ep-{episode}')

            epsilon *= epsilon_decay