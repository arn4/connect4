from deep_connect4.models import ConvolutionalNeuralNetwork
from deep_connect4.supervised import NeuralNetworkTrainer

from connect4.GameEngine import Tournament
from connect4.Players import CenteredAlphaBetaPlayer, TensorFlowProabilitiesPlayer, TwoStagePlayer, RandomPlayer

datasets = [
    'generated-datasets/noisy-5-.6_noisy-5-.6_centered-6_x10000.npz',
    'generated-datasets/noisy-4-1._noisy-4-1._centered-5_x1000.npz',
    'generated-datasets/random_random_centered-6_x1000.npz',
    'generated-datasets/random_random_centered-6_x15000.npz',
    'generated-datasets/noisy-4-1._noisy-4-1._centered-6_x15000.npz'
]

output_model = 'trained-models/supervised/only-simmetry'
initial_rand = 2


def randomize_player(player):
    return TwoStagePlayer(RandomPlayer(), player, initial_rand)


if __name__ == "__main__":
    nnt = NeuralNetworkTrainer(datasets, ConvolutionalNeuralNetwork, output_model)
    nnt.clean_dataset(drop_duplicates = False, reflection_simmetry = True, max_accepted_entropy = None)
    nnt.train(300, 1000, 3)

    t = Tournament(
        randomize_player(TensorFlowProabilitiesPlayer('trained-models/supervised/only-simmetry_ep-252', 'Supervised DNN')),
        randomize_player(CenteredAlphaBetaPlayer(6, name='Supervisor'))
    )
    t.play_games(1000)
    print(t)
