from deep_connect4.reinforcement import RLNeuralNetworkTrainer
from deep_connect4.models import ScoreConvolutionalNeuralNetwork

from connect4.GameEngine import Tournament
from connect4.Players import AlphaBetaPlayer, PoolPlayer, TensorFlowScorePlayer, RandomPlayer, TwoStagePlayer

output_model = 'trained-models/reinforcement/rl-50k-3players-mineps0.2-lindecay'
initial_rand = 2


def randomize_player(player):
    return TwoStagePlayer(RandomPlayer(), player, initial_rand)


if __name__ == '__main__':
    rlnn = RLNeuralNetworkTrainer(ScoreConvolutionalNeuralNetwork, 3, output_model)
    rlnn.train(
        episodes = 50000,
        epsilon_decay = 0.00002,
        batch_size = 200, min_epsilon = 0.2,
        decay_strategy = 'linear',
        use_symmetry = True,
        save_every = 1000
    )

    good_players = [(0, 9000), (1, 10000), (2, 20000)]
    t = Tournament(
        randomize_player(PoolPlayer([TensorFlowScorePlayer(output_model + f'_ep-{ep}_id-{id}') for id, ep in good_players], name='Pool of RL players')),
        randomize_player(AlphaBetaPlayer(3, name='CAB 3'))
    )
    t.play_games(1000)
    print(t)
