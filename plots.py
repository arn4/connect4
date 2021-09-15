
from connect4.GameEngine import Tournament
from connect4.Players import AlphaBetaPlayer, RandomPlayer, TwoStagePlayer
from connect4.costants import P1, P2
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool

output_directory = 'latex/img/'
comparator_player = AlphaBetaPlayer(3)
y_label = 'Wins against \\texttt{AlphaBetaPlayer(3)[\\si{\\percent}]}'
n_games_comparsion = 0
n_jobs = None

# Matplotlib style
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble='\\usepackage{amsmath} \\usepackage{siunitx}')
plt.style.use('seaborn')

## Supervised
supervised_path = 'trained-models/supervised/'


def get_supervised_player_score(player_path):
    from connect4.Players import TensorFlowProabilitiesPlayer
    t = Tournament(
        TwoStagePlayer(RandomPlayer(), TensorFlowProabilitiesPlayer(player_path), 2),
        TwoStagePlayer(RandomPlayer(), comparator_player, 2)
    )
    t.play_games(n_games_comparsion)
    return t.counter[P1][P1] + t.counter[P2][P1]


def supervised_plot(models_data, output_file):
    fig, ax = plt.subplots(figsize=(7,6))
    ax.set_xlabel('Epochs')
    ax.set_ylabel(y_label)
    pool = Pool(n_jobs)
    for md in models_data:
        players_path = [supervised_path + md[0].format(ep) for ep in md[1]]
        results = np.array(pool.map(get_supervised_player_score, players_path))/n_games_comparsion*100
        ax.plot(list(md[1]), results, label = md[2])
    ax.legend()
    fig.savefig(output_directory + output_file, format = 'pdf', bbox_inches = 'tight')



## Reinforcment
reinforcement_path = 'trained-models/reinforcement/'

def get_reinforcement_player_score(player_path):
    from connect4.Players import TensorFlowScorePlayer
    t = Tournament(
        TwoStagePlayer(RandomPlayer(), TensorFlowScorePlayer(player_path), 2),
        TwoStagePlayer(RandomPlayer(), comparator_player, 2)
    )
    t.play_games(n_games_comparsion)
    return t.counter[P1][P1] + t.counter[P2][P1]

def rl_plot(models_data, output_file):
    fig, ax = plt.subplots(figsize=(7,6))
    ax.set_xlabel('Episodes')
    ax.set_ylabel(y_label)
    pool = Pool(n_jobs)
    for md in models_data:
        players_paths = [reinforcement_path + md[0].format(ep) for ep in md[1]]
        results = np.array(pool.map(get_reinforcement_player_score, players_paths))/n_games_comparsion*100
        ax.plot(list(md[1]), results, label = md[2])
    ax.legend()
    fig.savefig(output_directory + output_file, format = 'pdf', bbox_inches = 'tight')

rl_models_data = [
    # ('rl-180k-single-mineps0.20_ep-{}_id-0', range(1000, 47001, 1000), 'Really long training')
    # ('rl-30k-3players-mineps0.20_ep-{}_id-0', range(400, 30001, 400), 'Agent 1'),
    # ('rl-30k-3players-mineps0.20_ep-{}_id-1', range(400, 30001, 400), 'Agent 2'),
    # ('rl-30k-3players-mineps0.20_ep-{}_id-2', range(400, 30001, 400), 'Agent 3'),
]

ep_10k = range(400, 10001, 400)
rl_10k_comparsion = [
    ('rl-10k-3players_ep-{}_id-0', ep_10k, 'Multiple Training - Agent 1'),
    ('rl-10k-3players_ep-{}_id-1', ep_10k, 'Multiple Training - Agent 2'),
    ('rl-10k-3players_ep-{}_id-2', ep_10k, 'Multiple Training - Agent 3'),
    ('rl-10k-single_ep-{}_id-0', ep_10k,  'Single'),
    ('rl-10k-single-linear-decay_ep-{}_id-0', ep_10k, 'Single - Lin. Decay'),
    ('rl-flatten-10k-single_ep-{}_id-0', ep_10k, 'Flatten NN'),
]

rl_50k = [
    ('rl-50k-3players-mineps0.2-lindecay_ep-{}_id-0', range(1000, 50001, 1000), 'Multiple Training - Agent 1'),
    ('rl-50k-3players-mineps0.2-lindecay_ep-{}_id-1', range(1000, 50001, 1000), 'Multiple Training - Agent 2'),
    ('rl-50k-3players-mineps0.2-lindecay_ep-{}_id-2', range(1000, 50001, 1000), 'Multiple Training - Agent 3')
]

ep_300_50 = range(50, 301, 50)
ep_300_3  = range(3, 301, 3)

supervised_models_data = [
    ('full-clean_ep-{}', ep_300_3, 'Drop low information + Drop Duplicates + Symmetry'),
    ('no-duplicates-simmetry_ep-{}', ep_300_3, 'Drop duplicates + Symmetry'),
    ('only-entropy_ep-{}', ep_300_3, 'Drop low information'),
    ('only-simmetry_ep-{}', ep_300_3, 'Simmetry'),
    ('raw_ep-{}', ep_300_3, 'Raw data')
]

if __name__ == '__main__':
    supervised_plot(supervised_models_data, 'supervised.pdf')
    rl_plot(rl_10k_comparsion, 'reinforcement-10k-comparsion.pdf')
    rl_plot(rl_50k, 'reinforcement-50k.pdf')