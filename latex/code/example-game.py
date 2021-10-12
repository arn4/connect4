from connect4.GameEngine import Game
from connect4.Players import NoisyAlphaBetaPlayer, PerfectPlayer

g = Game(
  NoisyAlphaBetaPlayer(depth = 7, noise = 0.5, name = 'Yukihiro Matsumoto'),
  PerfectPlayer('./perfect-player', name = 'Guido van Rossum'),
  starter = P2
)

while not g.finish:
  print(g)
  g.next()
print(g)