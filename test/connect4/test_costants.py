from connect4.costants import *

def test_minsizeboard():
    assert(N_COL >= 4)
    assert(N_ROW >= 4)

def test_costants():
    assert(P1 == 1)
    assert(P2 == 2)
    assert(NOBODY == 0)

def test_colors():
    assert(RED == 1)
    assert(YELLOW == 2)
