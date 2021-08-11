import tensorflow as tf
import numpy as np
import math
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D
from tensorflow.keras import Model
from scipy.stats import entropy

class NeuralNetworkTrainer():
    def __init__(self, raw_datasets, neural_network_model, model_name):
        boards_list = []
        moves_list = []

        for ds in raw_datasets:
            with np.load(ds) as data:
                boards_list.append(data['boards'])
                moves_list.append(data['moves'])

        self.boards = np.concatenate(boards_list)
        self.moves = np.concatenate(moves_list)
        assert(len(self.boards) == len(self.moves))
        print(f'Raw data dimension: {len(self.moves)}')

        self.model = neural_network_model()
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss = tf.keras.metrics.Mean(name='train_loss')
        self.metric = tf.keras.metrics.BinaryCrossentropy(name='train_metric')

        self.model_name = model_name

    def clean_dataset(self, drop_duplicates = True, reflection_simmetry = True, max_accepted_entropy = None):
        # Remove high entropy entries
        if max_accepted_entropy is not None:
            keep = [entropy(m) < max_accepted_entropy for m in self.moves]
            self.boards = self.boards[keep]
            self.moves = self.moves[keep]
            assert(len(self.boards) == len(self.moves))
        print(f'After Entropy: {len(self.moves)}')
        
        # Use the simmetry of the boards to increase test cases
        if reflection_simmetry:
            flipped_boards = np.flip(self.boards, 2)
            flipped_moves = np.flip(self.moves, 1)
            self.boards = np.concatenate((self.boards, flipped_boards))
            self.moves = np.concatenate((self.moves, flipped_moves))
            assert(len(self.boards) == len(self.moves))
        print(f'After reflection: {len(self.moves)}')

        # Drop duplicates
        if drop_duplicates:
            table = {b.tobytes():(b,[])  for b in self.boards}

            for b, m in zip(self.boards, self.moves):
                table[b.tobytes()][1].append(m)
            
            new_boards_list = []
            new_moves_list = []

            for b, bml in table.items():
                new_boards_list.append(bml[0])
                new_moves_list.append(sum(bml[1])/len(bml[1]))
                assert(math.isclose(sum(new_moves_list[-1]), 1.))
            
            self.boards = np.array(new_boards_list)
            self.moves  = np.array(new_moves_list)
        print(f'After duplicates: {len(self.moves)}')
            
    @tf.function
    def _train_step(self, board_batch, moves_batch):
        with tf.GradientTape() as tape:
            batch_predictions = self.model(board_batch, training=True)
            loss_on_batch = self.loss_object(moves_batch, batch_predictions)
        gradients = tape.gradient(loss_on_batch, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.loss(loss_on_batch)
        self.metric(moves_batch, batch_predictions)

    def train(self, epochs, batch_size, save_every = None):
        if save_every is None:
            save_every = epochs

        dataset = tf.data.Dataset.from_tensor_slices((self.boards, self.moves))
        dataset = dataset.shuffle(len(self.boards))
        dataset = dataset.batch(batch_size)

        for epoch in range(1, epochs + 1):
            # Reset the metrics at the start of the next epoch
            self.loss.reset_states()
            self.metric.reset_states()

            for boards_batch, moves_batch in dataset:
                self._train_step(boards_batch, moves_batch)

            print(
                f'Epoch {epoch}, '
                f'Loss: {self.loss.result()}, '
                f'Binary Crossentropy: {self.metric.result()}'
            )
            
            if epoch % save_every == 0:
                print(f'Saving {self.model_name}_ep-{epoch}')
                self.model.save(self.model_name + f'_ep-{epoch}')

