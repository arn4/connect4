from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D
from tensorflow.keras import Model
import tensorflow as tf

from connect4.costants import N_ROW, N_COL

class NeuralNetwork(Model):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = Flatten(input_shape=(N_ROW, N_COL))
        self.hidden1 = Dense(100, activation='relu')
        self.hidden2 = Dense(50, activation='relu')
        self.hidden3 = Dense(50, activation='relu')
        self.outlayer = Dense(N_COL)

    def call(self, x):
        x = self.flatten(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        return self.outlayer(x)

class ConvolutionalNeuralNetwork(Model):
    def __init__(self):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.conv = Conv2D(100, 4, input_shape=(N_ROW, N_COL))
        self.hidden1 = Dense(50, activation='relu')
        self.hidden2 = Dense(50, activation='relu')
        self.outlayer = Dense(N_COL)

    def call(self, x):
        x = self.conv(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        return self.outlayer(x)

class ScoreConvolutionalNeuralNetwork(Model):
    def __init__(self):
        super(ScoreConvolutionalNeuralNetwork, self).__init__()
        self.conv = Conv2D(150, kernel_size = 4, input_shape=(N_ROW, N_COL, 1), activation = 'relu')
        self.flatten = Flatten()
        self.hidden1 = Dense(100, activation='relu')
        self.hidden2 = Dense(100, activation='relu')
        self.outlayer = Dense(1)

    def call(self, x):
        x = tf.expand_dims(x, 3) # I need to had a channel dimemension to use Conv2D layer
        x = tf.cast(x, tf.float32) # I need to convert to an allowed dtype for convolutional network
        x = self.conv(x)
        x = self.flatten(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        return self.outlayer(x)

class ScoreNeuralNetwork(Model):
    def __init__(self):
        super(ScoreNeuralNetwork, self).__init__()
        self.flatten = Flatten(input_shape=(N_ROW, N_COL))
        self.hidden1 = Dense(150, activation='relu')
        self.hidden2 = Dense(100, activation='relu')
        self.hidden3 = Dense(100, activation='relu')
        self.outlayer = Dense(1)

    def call(self, x):
        x = self.flatten(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        return self.outlayer(x)