
from connect4.GameEngine import Tournament
from connect4.Players import AlphaBetaPlayer, RandomPlayer, TwoStagePlayer
from connect4.costants import P1, P2
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool

output_directory = 'latex/img'
comparator_player = AlphaBetaPlayer(3)
n_games_comparsion = 300
n_jobs = None

# Matplotlib style
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble='\\usepackage{amsmath} \\usepackage{siunitx}')
plt.style.use('seaborn')

## Supervised
supervised_path = 'trained-models/supervised'
n_epochs = list(range(50, 301, 50))
supervised_models = [
    'full-clean',
    'no-duplicates-simmetry',
    'only-entropy',
    'only-simmetry',
    'raw'
]

def get_supervised_player_score(player_path):
    from connect4.Players import TensorFlowProabilitiesPlayer
    t = Tournament(
        TwoStagePlayer(RandomPlayer(), TensorFlowProabilitiesPlayer(player_path), 2),
        TwoStagePlayer(RandomPlayer(), comparator_player, 2)
    )
    t.play_games(n_games_comparsion)
    return t.counter[P1][P1] + t.counter[P2][P1]


def supervised_plot():
    fig, ax = plt.subplots(figsize=(7,6))
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Wins against \\texttt{AlphaBetaPlayer(3)}')
    pool = Pool(n_jobs)
    for model in supervised_models:
        players_path = [f'{supervised_path}/{model}_ep-{ep}' for ep in n_epochs]
        results = np.array(pool.map(get_supervised_player_score, players_path))/n_games_comparsion*100
        ax.plot(n_episodes, results, label = model)
    ax.legend()
    fig.savefig(f'{output_directory}/supervised.pdf', format = 'pdf', bbox_inches = 'tight')



## Reinforcment
reinforcement_path = 'trained-models/reinforcement'
n_episodes = list(range(5000, 100001, 5000))
reinforcement_models = [
    '',
    # 'conv-',
]

def get_reinforcement_player_score(player_path):
    from connect4.Players import TensorFlowScorePlayer
    t = Tournament(
        TwoStagePlayer(RandomPlayer(), TensorFlowScorePlayer(player_path), 2),
        TwoStagePlayer(RandomPlayer(), comparator_player, 2)
    )
    t.play_games(n_games_comparsion)
    return t.counter[P1][P1] + t.counter[P2][P1]

def rlsingle_plot():
    fig, ax = plt.subplots(figsize=(7,6))
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Wins against \\texttt{AlphaBetaPlayer(3)[\\si{\\percent}]}')
    pool = Pool(n_jobs)
    for model in reinforcement_models:
        # players_path = [f'{reinforcement_path}/rl-{model}100000-0.99996_ep-{ep}_id-0' for ep in n_episodes]
        players_path = [f'{reinforcement_path}/rl-{model}100000-0.99995-sym_ep-{ep}' for ep in n_episodes]
        results = np.array(pool.map(get_reinforcement_player_score, players_path))/n_games_comparsion*100
        ax.plot(n_episodes, results, label = model)
    ax.legend()
    fig.savefig(f'{output_directory}/reinforcement-single.pdf', format = 'pdf', bbox_inches = 'tight')


if __name__ == '__main__':
    # supervised_plot()
    rlsingle_plot()