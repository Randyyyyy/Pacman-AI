from Classes import *
import pygame
import time
import numpy as np
import sys
import argparse

'''mode: manual, QLearning, SARSA, ApproxQ'''
'''gridName: miniGrid, xsmallGrid, smallGrid, mediumGrid, largeGrid'''


'''only manual and ApproxQ work for capsule atm'''

if __name__ == "__main__" or __name__ == "Pacman":
    pygame.init()

    game = Game(epsilon=0.05, gamma=0.99, lr=0.01,
                trainEpi=200, trainDelay=1, testEpi=100, testDelay=1,
                mode="ApproxQ", gridName="largeGrid")
    pacman = Pacman(game)
    # ghosts = [Ghost(game, "GREEN")]
    ghosts = [Ghost(game, "GREEN"), Ghost(game, "RED")]
    # ghosts = [Ghost(game, "GREEN"), Ghost(game, "RED"), Ghost(game, "PINK")]

    run = Run(game, pacman, ghosts)
    run.flow()

    # run.MCST_Train(mode=False)