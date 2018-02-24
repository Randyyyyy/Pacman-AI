import pygame
from pygame.locals import *
from copy import deepcopy
import heapq as hq
import time
import numpy as np
from random import randint
from collections import Counter
import matplotlib.pyplot as plt

'''
QL: 
(state, action, reward, nextstate)

'''

class Game():

    def __init__(self, epsilon, gamma, lr, trainEpi, trainDelay, testEpi, testDelay, mode, gridName):
        # e-greedy, discount factor, learning rate
        self.epsilon, self.gamma, self.lr = epsilon, gamma, lr
        # training and testing episodes
        self.trainEpi, self.testEpi = trainEpi, testEpi
        # running mode
        '''ADDRESS MODE SELECTION'''
        self.mode = mode
        self.trainDelay = trainDelay
        self.testDelay = testDelay

        # game knowledge
        self.grid, self.gridName = [], gridName
        self.oldScore = 0
        self.score = 0
        self.episode = 1
        self.foodGrid = []
        self.walls = []
        self.paths = []
        self.capsule = []
        self.won = self.lost = False

        # display settings
        self.font = pygame.font.Font(None, 30)
        self.font = pygame.font.SysFont("ubunturegular", 30)
        self.BLACK = (0, 0, 0)
        self.YELLOW = (249, 230, 84)
        self.GREY = (50, 50, 50)
        self.RED = (214, 70, 70)
        self.BLUE = (84, 136, 249)
        self.WHITE = (255, 255, 255)
        self.GREEN = (55, 173, 90)
        self.PINK = (255, 188, 188)
        # the game board
        self.size = [600, 400]
        self.width = 20
        self.height = 20
        self.margin = 1
        self.create()
        self.screen = pygame.display.set_mode(self.size)

    def create(self):
        miniGrid = [[1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 1],
                    [1, 0, 1, 0, 1],
                    [1, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1]]
        xsmallGrid = [[1, 1, 1, 1, 1, 1],
                      [1, 0, 0, 0, 0, 1],
                      [1, 0, 1, 1, 0, 1],
                      [1, 0, 1, 0, 0, 1],
                      [1, 0, 1, 1, 0, 1],
                      [1, 0, 0, 0, 0, 1],
                      [1, 1, 1, 1, 1, 1]]
        smallGrid = [[1, 1, 1, 1, 1, 1, 1],
                     [1, 0, 0, 0, 0, 0, 1],
                     [1, 0, 1, 0, 1, 0, 1],
                     [1, 0, 0, 0, 0, 0, 1],
                     [1, 0, 1, 1, 1, 0, 1],
                     [1, 0, 0, 0, 0, 0, 1],
                     [1, 1, 1, 1, 1, 1, 1]]
        mediumGrid = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                      [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
                      [1, 0, 1, 0, 1, 0, 0, 0, 0, 1],
                      [1, 0, 1, 0, 1, 1, 1, 1, 0, 1],
                      [1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                      [1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
                      [1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
                      [1, 0, 0, 0, 0, 0, 0, 0, 2, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        largeGrid = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 1, ],
                     [1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, ],
                     [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, ],
                     [1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 1, ],
                     [1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, ],
                     [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, ],
                     [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, ],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, ],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ]
                     ]
        gridList = {"miniGrid": miniGrid,
                    "xsmallGrid": xsmallGrid,
                    "smallGrid": smallGrid,
                    "mediumGrid": mediumGrid,
                    "largeGrid": largeGrid}
        self.grid = gridList[self.gridName]

        # grid[y, x]
        yA = len(self.grid)
        xA = len(self.grid[0])
        self.foodGrid = np.zeros((yA, xA))
        self.capsule = np.zeros((yA, xA))
        for y in range(yA):
            for x in range(xA):
                if self.grid[y][x] == 1:
                    # meaning it's a wall and insert it into wall list
                    self.walls.append((x, y))  # (x, y)
                elif self.grid[y][x] == 0:
                    # food
                    self.foodGrid[y, x] = 1
                    self.paths.append((x, y))
                elif self.grid[y][x] == 2:
                    # capsule
                    self.capsule[y, x] = 1

        # special simple case: only 1 food
        if self.gridName == "xsmallGrid":
            self.foodGrid = np.zeros((yA, xA))
            self.foodGrid[int(yA/2), int(xA/2)] = 1
        return self

    def reset(self):
        self.grid = []
        self.walls = []
        self.score = 0
        self.oldScore = 0
        self.won = self.lost = False
        self.create()
        return self

    def timeScore(self):
        self.score -= 1

    def scoreShow(self):
        scoretext = self.font.render("Score: " + (str)(self.score), 1, self.RED)
        self.screen.blit(scoretext, (30, 350))

    def episodeShow(self):
        episodeText = self.font.render("Episode: " + (str)(self.episode), 1, self.RED)
        self.screen.blit(episodeText, (250, 350))

    '''show training or testing'''
    def phaseShow(self, state):
        stateText = self.font.render((str)(state), 1, self.RED)
        self.screen.blit(stateText, (470, 350))

    # update map and food
    def mapShow(self):
        xA = len(self.grid[0])
        yA = len(self.grid)
        for x in range(xA):
            for y in range(yA):
                # bg color
                pygame.draw.rect(self.screen, self.GREY, [(self.margin + self.width) * x + self.margin,
                                                      (self.margin + self.height) * y + self.margin, self.width,
                                                      self.height])
                # food color
                if self.foodGrid[y][x] == 1:
                    # paint food
                    pygame.draw.circle(self.screen, self.WHITE,
                                       [(self.margin + self.width) * x + self.margin + (int)(self.width / 2),
                                        (self.margin + self.height) * y + self.margin + (int)(self.height / 2)],
                                       (int)((self.width - 14) / 2))

                if self.capsule[y][x] == 1:
                    # paint capsule
                    pygame.draw.circle(self.screen, self.WHITE,
                                       [(self.margin + self.width) * x + self.margin + (int)(self.width / 2),
                                        (self.margin + self.height) * y + self.margin + (int)(self.height / 2)],
                                       (int)((self.width - 7) / 2))
        # draw walls
        for wall in self.walls:
            pygame.draw.rect(self.screen, self.BLUE,
                             [(self.margin + self.width) * wall[0] + self.margin,
                              (self.margin + self.height) * wall[1] + self.margin,
                              self.width, self.height])

    '''check if episode ends and change lost/won'''
    def terminal(self, pacman, ghosts):
        if self.foodGrid.sum() == 0:
            self.won = True
            return True
        else:
            for ghost in ghosts:
                if not ghost.scared and (pacman.x, pacman.y) == (ghost.x, ghost.y):
                    self.lost = True
                    return True
        return False

    '''when terminal, update score and print message'''
    def terScore(self):
        if self.lost:
            self.score -= 1000
            print((str)(self.episode) + " Lost! Final Score = " + (str)(self.score))
        elif self.won:
            self.score += 100
            print((str)(self.episode) + " Won! Final Score = " + (str)(self.score))

    def terminal_BU(self):
        if self.lost:
            print((str)(self.episode) + " Lost! Final Score = " + (str)(self.score))
            return True
        elif self.won:
            print((str)(self.episode) + " Won! Final Score = " + (str)(self.score))
            return True
        return False

class Agent():

    def __init__(self, x, y, game):
        self.x = x
        self.y = y
        self.game = game

    def isWall(self, x, y):
        if (x, y) in self.game.walls:
            return True
        else:
            return False

    def getPosition(self, x, y, action):
        if action != None:
            if action == 0: return (x-1, y)
            elif action == 1: return (x+1, y)
            elif action == 2: return (x, y-1)
            elif action == 3: return (x, y+1)
        return (x, y)

    def getLegalActions(self, x, y):
        '''0, 1, 2, 3 for left, right, up, down'''
        list = []
        if self.isWall(x - 1, y) == False: list.append(0)
        if self.isWall(x + 1, y) == False: list.append(1)
        if self.isWall(x, y - 1) == False: list.append(2)
        if self.isWall(x, y + 1) == False: list.append(3)
        return list

    def getLegalNeighbors(self, x, y):
        list = []
        if self.isWall(x - 1, y) == False: list.append((x - 1, y))
        if self.isWall(x + 1, y) == False: list.append((x + 1, y))
        if self.isWall(x, y - 1) == False: list.append((x, y - 1))
        if self.isWall(x, y + 1) == False: list.append((x, y + 1))
        return list

    def takeLegalAction(self, action):
        if action != None:
            if action == 0: self.x = self.x - 1
            if action == 1: self.x = self.x + 1
            if action == 2: self.y = self.y - 1
            if action == 3: self.y = self.y + 1
        # return self

    def getDistance(self, xs, ys, xd, yd):
        return abs(xs - xd) + abs(ys - yd)

class Ghost(Agent):

    def __init__(self, game, color):
        self.game = game
        self.action = randint(0, 3)
        self.color = color
        self.colorOri = color   #original color
        self.scared = False
        self.scareTimerAll = 100
        self.scareTimer = self.scareTimerAll
        grid = game.grid
        xA = len(grid[0])
        yA = len(grid)
        # spawn at center of the map
        x = (int)(xA / 2)
        y = (int)(yA / 2)
        if game.gridName == "miniGrid":
            x, y = xA - 2, yA - 2
        Agent.__init__(self, x, y, game)

    def reset(self):
        self.scared = False
        self.color = self.colorOri
        self.scareTimerAll = 40
        self.scareTimer = self.scareTimerAll

        grid = self.game.grid
        xA = len(grid[0])
        yA = len(grid)
        # spawn at center of the map
        x = (int)(xA / 2)
        y = (int)(yA / 2)
        if self.game.gridName == "miniGrid":
            x, y = xA - 2, yA - 2
        Agent.__init__(self, x, y, self.game)

    def show(self):
        game = self.game
        color = game.RED
        if self.color == "GREEN":
            color = game.GREEN
        if self.color == "WHITE":
            color = game.WHITE
        if self.color == "PINK":
            color = game.PINK
        pygame.draw.circle(game.screen, color,
                           [(game.margin + game.width) * self.x + game.margin + (int)(game.width / 2),
                            (game.margin + game.height) * self.y + game.margin + (int)(game.height / 2)],
                           (int)(game.width / 2), game.margin*4)

    def getReverseAction(self, action):
        if action == 0 or action == 1: return 1-action
        if action == 2 or action == 3: return 5-action

    def updateScared(self):
        if self.scared:
            if self.color != "WHITE":
                self.colorOri = self.color
                self.color = "WHITE"
            self.scareTimer -= 1

        if self.scareTimer == 0:
            self.scared = False
            self.color = self.colorOri
            self.scareTimer = self.scareTimerAll

    def move(self):
        aList = self.getLegalActions(self.x, self.y)

        if len(aList) == 1:
            self.action = aList[0]
        elif len(aList) > 2 or len(aList) <= 2 and self.action not in aList:
            '''remove backward action'''
            rev = self.getReverseAction(self.action)
            if rev in aList:
                aList.remove(rev)
            index = randint(0, len(aList) - 1)
            self.action = aList[index]
            # otherwise same action
        # act
        self.takeLegalAction(self.action)
        # update scared status if scared
        self.updateScared()

class Pacman(Agent):

    def __init__(self, game):
        self.game = game

        self.epsilon = game.epsilon
        self.gamma = game.gamma
        self.lr = game.lr
        self.reward = 0
        self.Q = Counter()
        self.weight = Counter()

        x, y = 1, 1
        Agent.__init__(self, x, y, game)

    def reset(self):
        paths = self.game.paths
        x, y = 1, 1
        # if self.game.gridName == "largeGrid":
        # #     index = randint(0, len(paths)-1)
        # #     x, y = paths[index]
        # #
        # # elif self.game.gridName == "miniGrid" or self.game.gridName == "xsmallGrid":
        # #         x, y = 1, 1
        #
        # # else:
        #     grid = self.game.grid
        #     xA = len(grid[0])
        #     yA = len(grid)
        #     start = [(1, 1),
        #              (1, yA - 2),
        #              (xA - 2, 1),
        #              (xA - 2, yA - 2)]
        #     index = randint(0, len(start) - 1)
        #     x, y = start[index]

        Agent.__init__(self, x, y, self.game)

    '''transform list of lists to tuple of strings'''
    def getTS(self, input):
        out = []
        for row in input:
            out.append([])
            for item in row:
                if item == 1:
                    out[-1].append("1")
                else:
                    out[-1].append("0")
            out[-1] = ''.join(out[-1])
        out = tuple(out)
        return out

    '''get the state'''
    '''pac, ghosts, food, wall'''
    def getState(self, ghosts):
        '''state pac'''
        foodGrid = self.game.foodGrid
        yT, xT = len(foodGrid), len(foodGrid[0])
        pac = np.zeros((yT, xT))
        pac[self.y, self.x] = 1
        sPac = self.getTS(pac)

        '''state ghosts'''
        gho = np.zeros((yT, xT))
        for ghost in ghosts:
            gho[ghost.y, ghost.x] = 1
        sGho = self.getTS(gho)

        '''state food'''
        sFood = self.getTS(foodGrid)

        '''state wall'''
        sWall = self.getTS(self.game.grid)

        return (sPac, sGho, sFood, sWall)

    def hasFood(self, x, y):
        if self.game.foodGrid[y][x] == 1:
            return True
        else:
            return False

    def eatFood(self, x, y):
        self.game.foodGrid[y][x] = 0
        self.game.score += 5

    def hasCapsule(self, x, y):
        return self.game.capsule[y, x] == 1

    def eatCapsule(self, x, y):
        self.game.capsule[y][x] = 0
        self.game.score += 7

    def hasGhost(self, x, y, ghosts):
        for ghost in ghosts:
            if not ghost.scared and (ghost.x, ghost.y) == (x, y):
                return True
        return False

    def hasOneScaredGhost(self, x, y, ghost):
        return ghost.scared and (ghost.x, ghost.y) == (x, y)

    def countAllScaredGhosts(self, ghosts):
        count = 0
        for ghost in ghosts:
            if ghost.scared:
                count += 1
        return count

    def eatGhost(self):
        self.game.score += 50

    def show(self):
        game = self.game
        pygame.draw.circle(game.screen, game.YELLOW,
                           [(game.margin + game.width) * self.x + game.margin + (int)(game.width / 2),
                            (game.margin + game.height) * self.y + game.margin + (int)(game.height / 2)],
                           (int)(game.width / 2))

    def getReward(self):
        diff = self.game.score - self.game.oldScore

        if diff > 20:   # eat ghost
            self.reward = 50
        elif diff > 10: # eat capsule
            self.reward = 7
        elif diff > 0:    # eat food
            self.reward = 5
        elif diff < -10:    # killed by ghost
            self.reward = -1000
        else:           # elapsed time
            self.reward = -1

        if self.game.terminal and self.game.won:
            self.reward = 100

        self.game.oldScore = self.game.score

    '''approx Q helpers'''
    def nearestFood(self, x, y):
        '''search nearest food, return distance'''
        food = self.game.foodGrid

        # tuples of x, y, distance
        proximity = []
        visited = set()

        hq.heappush(proximity, (0, x, y))
        while len(proximity) != 0:
            d, x, y = hq.heappop(proximity)
            sList = self.getLegalNeighbors(x, y)
            for s in sList:
                # visited
                if s in visited:
                    continue
                px, py = s
                # has food: return distance
                if food[py][px] == 1:
                    return d+1
                # else: mark visited and push all neighbors
                visited.add((px, py))
                hq.heappush(proximity, (d+1, px, py))
                # nList = self.getLegalNeighbors(px, py)
                # for neighbor in nList:
                #     nbrx, nbry = neighbor
                #     hq.heappush(proximity, (d+2, nbrx, nbry))
                #     # proximity.append((nbrx, nbry, d+1))
        # no food
        return None

    def nearestCapsule(self, x, y):
        '''search nearest food, return distance'''
        capsule = self.game.capsule

        # tuples of x, y, distance
        proximity = []
        visited = set()

        hq.heappush(proximity, (0, x, y))
        while len(proximity) != 0:
            d, x, y = hq.heappop(proximity)
            sList = self.getLegalNeighbors(x, y)
            for s in sList:
                # visited
                if s in visited:
                    continue
                px, py = s
                # has food: return distance
                if capsule[py][px] == 1:
                    return d+1
                # else: mark visited and push all neighbors
                visited.add((px, py))
                hq.heappush(proximity, (d+1, px, py))
                # nList = self.getLegalNeighbors(px, py)
                # for neighbor in nList:
                #     nbrx, nbry = neighbor
                #     hq.heappush(proximity, (d+2, nbrx, nbry))
                #     # proximity.append((nbrx, nbry, d+1))
        # no food
        return None

    def ghostDistance(self, x, y, ghosts):
        '''search nearest food, return distance'''

        # tuples of x, y, distance
        proximity = []
        visited = set()

        hq.heappush(proximity, (0, x, y))
        while len(proximity) != 0:
            d, x, y = hq.heappop(proximity)
            sList = self.getLegalNeighbors(x, y)
            for s in sList:
                # visited
                if s in visited:
                    continue
                px, py = s
                # has ghost
                for ghost in ghosts:
                    if s == (ghost.x, ghost.y):
                        return 1/(d+10)
                # else: mark visited and push all neighbors
                visited.add((px, py))
                hq.heappush(proximity, (d+1, px, py))
                # nList = self.getLegalNeighbors(px, py)
                # for neighbor in nList:
                #     nbrx, nbry = neighbor
                #     hq.heappush(proximity, (d+2, nbrx, nbry))
                #     # proximity.append((nbrx, nbry, d+1))
        # no ghosts
        return None

    def nearestGhost(self, x, y, k, ghosts):
        '''number of ghosts k step away'''

        # tuples of x, y, distance
        proximity = []
        visited = set()

        hq.heappush(proximity, (0, x, y))
        count = 0

        while len(proximity) != 0:
            d, x, y = hq.heappop(proximity)
            if d > k:
                break
            sList = self.getLegalNeighbors(x, y)
            for s in sList:
                # visited
                if s in visited:
                    continue
                px, py = s
                # has food: return distance

                for ghost in ghosts:
                    if s == (ghost.x, ghost.y) and not ghost.scared:
                        count += 1

                # else: mark visited and push all neighbors
                visited.add((px, py))
                hq.heappush(proximity, (d + 1, px, py))

        return count

    def nearestScaredGhost(self, x, y, k, ghosts):
        '''number of ghosts k step away'''

        # tuples of x, y, distance
        proximity = []
        visited = set()

        hq.heappush(proximity, (0, x, y))
        count = 0

        while len(proximity) != 0:
            d, x, y = hq.heappop(proximity)
            if d > k:
                break
            sList = self.getLegalNeighbors(x, y)
            for s in sList:
                # visited
                if s in visited:
                    continue
                px, py = s
                # has food: return distance

                for ghost in ghosts:
                    if s == (ghost.x, ghost.y) and ghost.scared:
                        count += 1

                # else: mark visited and push all neighbors
                visited.add((px, py))
                hq.heappush(proximity, (d + 1, px, py))

        return count

    def nearestFood_BU(self, x, y):
        '''search nearest food, return distance'''
        food = self.game.foodGrid

        # tuples of x, y, distance
        proximity = []
        visited = set()

        hq.heappush(proximity, (0, x, y))
        while len(proximity) != 0:
            d, x, y = hq.heappop(proximity)
            sList = self.getLegalNeighbors(x, y)
            for s in sList:
                # visited
                if s in visited:
                    continue
                px, py = s
                # has food: return distance
                if food[py][px] == 1:
                    return d+1
                # else: mark visited and push all neighbors
                visited.add((px, py))
                nList = self.getLegalNeighbors(px, py)
                for neighbor in nList:
                    nbrx, nbry = neighbor
                    hq.heappush(proximity, (d+2, nbrx, nbry))
                    # proximity.append((nbrx, nbry, d+1))
        # no food
        return None

    def nearestFoodCdn(self, x, y):
        '''nearest food coordinate'''
        food = self.game.foodGrid

        # tuples of x, y, distance
        proximity = []
        visited = set()

        hq.heappush(proximity, (0, x, y))
        while len(proximity) != 0:
            d, x, y = hq.heappop(proximity)
            sList = self.getLegalNeighbors(x, y)
            for s in sList:
                # visited
                if s in visited:
                    continue
                px, py = s
                # has food: return distance
                if food[py][px] == 1:
                    return (px, py)
                # else: mark visited and push all neighbors
                visited.add((px, py))
                nList = self.getLegalNeighbors(px, py)
                for neighbor in nList:
                    nbrx, nbry = neighbor
                    hq.heappush(proximity, (d+2, nbrx, nbry))
                    # proximity.append((nbrx, nbry, d+1))
        # no food
        return None

    def ghostBlock(self, x, y, a):
        '''find if a ghost block this direction'''
        

    def isCorner(self, x, y):
        '''if (x, y) is corner case'''
        aList = self.getLegalActions(x, y)
        avai = len(aList)
        for a in aList:
            if self.ghostBlock(x, y, a):
                avai -= 1
        return avai == 0

    def ghostDir(self, x, y, ghosts):
        gList = np.zeros(4)
        # tuples of x, y, distance
        proximity = []
        visited = set()

        hq.heappush(proximity, (0, x, y))
        while len(proximity) != 0:
            d, x, y = hq.heappop(proximity)
            sList = self.getLegalNeighbors(x, y)
            for s in sList:
                # visited
                if s in visited:
                    continue
                px, py = s

                for ghost in ghosts:
                    if (ghost.x, ghost.y) == s and d+1 <= 2:
                        if abs(px-ghost.x) > abs(py-ghost.y):
                            # left
                            if px-ghost.x > 0:
                                gList[0] = 1
                            # right
                            else:
                                gList[1] = 1
                        else:
                            # up
                            if py - ghost.y > 0:
                                gList[2] = 1
                            # right
                            else:
                                gList[3] = 1

                # else: mark visited and push all neighbors
                visited.add((px, py))
                nList = self.getLegalNeighbors(px, py)
                for neighbor in nList:
                    nbrx, nbry = neighbor
                    hq.heappush(proximity, (d + 2, nbrx, nbry))
                    # proximity.append((nbrx, nbry, d+1))

        return gList

    def getFeatures_ghostDirection(self, a, ghosts):

        '''return features Hashmap'''
        features = Counter()
        foodGrid = self.game.foodGrid
        yA, xA = len(foodGrid), len(foodGrid[0])

        x, y = self.getPosition(self.x, self.y, a)

        cdn = self.nearestFoodCdn(x, y)
        if cdn is not None:
            # policy divergence reduction
            xt, yt = cdn
            if abs(x - xt) > abs(y - yt):
                # left
                if x - xt > 0:
                    f = 0
                # right
                else:
                    f = 1
            else:
                # up
                if y - yt > 0:
                    f = 2
                # right
                else:
                    f = 3
            features["target-direction"] = f

        '''close ghost directions'''
        gList = self.ghostDir(x, y, ghosts)
        features["left-ghost"] = gList[0]
        features["right-ghost"] = gList[1]
        features["up-ghost"] = gList[2]
        features["down-ghost"] = gList[3]


        for f in features.values():
            # f /= float(len(features))
            f /= 10.0
        return features

    def getFeatures_OK(self, a, ghosts):

        '''return features Hashmap'''
        features = Counter()
        foodGrid = self.game.foodGrid
        yA, xA = len(foodGrid), len(foodGrid[0])

        # next location
        x, y = self.getPosition(self.x, self.y, a)

        '''bias'''
        features["bias"] = 1.0

        '''ghosts one step away otherwise food'''
        # ghost-1-step
        gCount = 0
        sList = self.getLegalNeighbors(x, y)
        sList.append((x, y))
        for ghost in ghosts:
            if (ghost.x, ghost.y) in sList:
                gCount += 1
        features["ghost-1-step"] = float(gCount)


        # otherwise if has-food
        if not features["ghost-1-step"] and foodGrid[y, x] == 1:
            features["has-food"] = 1.0

        '''nearest food distance'''
        d = self.nearestFood(x, y)
        if d is not None:
            # policy divergence reduction
            features["nearest-food"] = float(d / (xA+yA))
            # features["nearest-food"] = float(d)


        for f in features.values():
            # f /= float(len(features))
            f /= 10.0
        return features

    def getFeatures(self, a, ghosts):

        '''return features Hashmap'''
        features = Counter()
        foodGrid = self.game.foodGrid
        capsule = self.game.capsule
        yA, xA = len(foodGrid), len(foodGrid[0])

        # next location
        x, y = self.getPosition(self.x, self.y, a)

        '''bias'''
        features["bias"] = 1.0

        '''ghosts one step away otherwise food'''
        # ghost-1-step
        gCount = 0
        sList = self.getLegalNeighbors(x, y)
        sList.append((x, y))
        for ghost in ghosts:
            if (ghost.x, ghost.y) in sList and not ghost.scared:
                gCount += 1
        features["ghost-1-step"] = float(gCount)

        # otherwise...
        if not features["ghost-1-step"]:

            # otherwise if has-food
            if foodGrid[y, x] == 1:
                features["food-1-step"] = 1.0

            # otherwise if has scared ghosts
            sgCount = 0
            for ghost in ghosts:
                if (ghost.x, ghost.y) in sList and not ghost.scared:
                    sgCount += 1
            features["scared-ghost-1-step"] = float(sgCount)

            # otherwise if no scared ghosts and has capsule
            if capsule[y, x] == 1 and self.countAllScaredGhosts(ghosts) == 0:
                features["capsule-1-step"] = 1.0

        '''nearest food distance'''
        d = self.nearestFood(x, y)
        if d is not None:
            # policy divergence reduction
            features["nearest-food"] = float(d / (xA+yA))
            # features["nearest-food"] = float(d)
        #
        # '''number of ghost k step away'''
        features["ghost-k-step"] = float(self.nearestGhost(x, y, 3, ghosts))

        '''________Others___________'''
        # # otherwise if has-food
        # if not features["ghost-1-step"] and foodGrid[y, x] == 1:
        #     features["food-1-step"] = 1.0
        #
        # # otherwise if has-capsule
        # if not features["ghost-1-step"] and capsule[y, x] == 1:
        #     features["capsule-1-step"] = 1.0

        # # otherwise if scared ghosts k steps around
        # if self.countAllScaredGhosts(ghosts) > 0:
        #     features["scared-ghost-k-step"] = float(0.1*self.nearestScaredGhost(x, y, 1, ghosts))

        # '''nearest capsule distance'''
        # cd = self.nearestCapsule(x, y)
        # if cd is not None:
        #     # policy divergence reduction
        #     features["nearest-capsule"] = float(cd / (xA + yA))

        # '''number of scared ghost k step away'''
        # if self.countAllScaredGhosts(ghosts) > 0:
        #     features["scared-ghost-k-step"] = float(self.nearestScaredGhost(x, y, 3, ghosts))

        # '''1/(1+d) distance to nearest ghost'''
        # gd = self.ghostDistance(x, y, ghosts)
        # if gd is not None:
        #     features["ghost-distance"] = gd




        for f in features.values():
            # f /= float(len(features))
            f /= 10.0
        return features

    def getFeatures_(self, a, ghosts):

        '''return features Hashmap'''
        features = Counter()
        foodGrid = self.game.foodGrid
        capsule = self.game.capsule
        yA, xA = len(foodGrid), len(foodGrid[0])

        # next location
        x, y = self.getPosition(self.x, self.y, a)

        '''bias'''
        features["bias"] = 1.0

        '''ghosts one step away otherwise food'''
        # ghost-1-step
        gCount = 0
        sList = self.getLegalNeighbors(x, y)
        sList.append((x, y))
        for ghost in ghosts:
            if (ghost.x, ghost.y) in sList and not ghost.scared:
                gCount += 1
        features["ghost-1-step"] = float(gCount)

        # otherwise...
        if not features["ghost-1-step"]:
            # otherwise if has scared ghosts
            sgCount = 0
            for ghost in ghosts:
                if (ghost.x, ghost.y) in sList and not ghost.scared:
                    sgCount += 1
            features["scared-ghost-1-step"] = float(sgCount)

            # otherwise if has-food
            if foodGrid[y, x] == 1:
                features["food-1-step"] = 1.0

            # otherwise if no scared ghosts and has capsule
            if capsule[y, x] == 1 and self.countAllScaredGhosts(ghosts) == 0:
                features["capsule-1-step"] = 1.0
                '''number of scared ghost k step away'''
                features["scared-ghost-k-step"] = float(self.nearestScaredGhost(x, y, 3, ghosts))

        # # otherwise if has-food
        # if not features["ghost-1-step"] and foodGrid[y, x] == 1:
        #     features["food-1-step"] = 1.0
        #
        # # otherwise if has-capsule
        # if not features["ghost-1-step"] and capsule[y, x] == 1:
        #     features["capsule-1-step"] = 1.0

        '''nearest food distance'''
        d = self.nearestFood(x, y)
        if d is not None:
            # policy divergence reduction
            features["nearest-food"] = float(d / (xA+yA))
            # features["nearest-food"] = float(d)

        '''number of ghost k step away'''
        features["ghost-k-step"] = float(self.nearestGhost(x, y, 3, ghosts))

        # '''1/(1+d) distance to nearest ghost'''
        # gd = self.ghostDistance(x, y, ghosts)
        # if gd is not None:
        #     features["ghost-distance"] = gd




        for f in features.values():
            # f /= float(len(features))
            f /= 10.0
        return features

    def getFeatures_Double(self, a, ghosts):

        '''return features Hashmap'''
        features = Counter()
        foodGrid = self.game.foodGrid
        yA, xA = len(foodGrid), len(foodGrid[0])

        # next location
        x, y = self.getPosition(self.x, self.y, a)

        '''bias'''
        features["bias"] = 1.0

        '''state s: ghosts one step away otherwise food'''
        # ghost-1-step
        gCount = 0
        sList = self.getLegalNeighbors(self.x, self.y)

        for ghost in ghosts:
            if (ghost.x, ghost.y) in sList:
                gCount += 1
        features["s-ghost-1-step"] = float(gCount)
        # otherwise if has-food
        if not features["s-ghost-1-step"] and foodGrid[self.y, self.x] == 1:
            features["s-has-food"] = 1.0

        '''state sn: ghosts one step away otherwise food'''
        # ghost-1-step
        gCount = 0
        sList = self.getLegalNeighbors(x, y)
        sList.append((x, y))
        for ghost in ghosts:
            if (ghost.x, ghost.y) in sList:
                gCount += 1
        features["ghost-1-step"] = float(gCount)
        # otherwise if has-food
        if not features["ghost-1-step"] and foodGrid[y, x] == 1:
            features["has-food"] = 1.0

        '''state s: nearest food distance'''
        d = self.nearestFood(self.x, self.y)
        if d is not None:
            # policy divergence reduction
            features["s-nearest-food"] = float(d / (xA+yA))
            # features["nearest-food"] = float(d)

        '''state sn: nearest food distance'''
        d = self.nearestFood(x, y)
        if d is not None:
            # policy divergence reduction
            features["nearest-food"] = float(d / (xA + yA))
            # features["nearest-food"] = float(d)


        for f in features.values():
            # f /= float(len(features))
            f /= 10.0
        return features

    def QFromFeatures(self, features):
        '''get Q from features and weights'''
        Qs = 0
        for f in features.keys():
            Qs += features[f] * self.weight[f]
        return Qs

    '''____________Pacman Ctrl Algorithms____________'''

    def manual(self, ghosts):
        if self.game.terminal(self, ghosts):
            return True

        for ghost in ghosts:
            if self.hasOneScaredGhost(self.x, self.y, ghost):
                # when ghost is eaten, score counts and ghost reset
                self.eatGhost()
                ghost.reset()

        keys = pygame.key.get_pressed()
        aList = self.getLegalActions(self.x, self.y)

        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            act = 0
            if act in aList:
                self.takeLegalAction(act)
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            act = 1
            if act in aList:
                self.takeLegalAction(act)
        elif keys[pygame.K_UP] or keys[pygame.K_w]:
            act = 2
            if act in aList:
                self.takeLegalAction(act)
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            act = 3
            if act in aList:
                self.takeLegalAction(act)

        if self.hasFood(self.x, self.y):
            self.eatFood(self.x, self.y)

        if self.hasCapsule(self.x, self.y):
            # eat capsule and turn ghosts scared
            self.eatCapsule(self.x, self.y)
            for ghost in ghosts:
                ghost.scared = True
                ghost.scareTimer = ghost.scareTimerAll

        for ghost in ghosts:
            if self.hasOneScaredGhost(self.x, self.y, ghost):
                # when ghost is eaten, score counts and ghost reset
                self.eatGhost()
                ghost.reset()

        return self.game.terminal(self, ghosts)

    def QLearning(self, ghosts):

        '''check terminal before Pacman moves'''
        '''if terminal, quit'''
        ter =  self.game.terminal(self, ghosts)
        if ter:
            self.game.terScore()
            return True

        '''get current state'''
        s = self.getState(ghosts)

        '''get legal actions'''
        aList = self.getLegalActions(self.x, self.y)

        Qlist = []
        for action in aList:
            Qlist.append(self.Q[(s, action)])

        '''e-greedy'''
        pr = np.ones(len(Qlist))*self.epsilon/len(Qlist)
        pr[np.argmax(Qlist)] += 1 - self.epsilon
        a = np.random.choice(aList, p=pr)

        '''act'''
        self.takeLegalAction(a)
        if self.hasFood(self.x, self.y):
            self.eatFood(self.x, self.y)
        '''check terminal'''
        ter = self.game.terminal(self, ghosts)
        if ter:
            self.game.terScore()
        '''get reward'''
        self.getReward()

        '''get next state'''
        sn = self.getState(ghosts)

        '''max Qsn'''
        # get legal actions
        list = self.getLegalActions(self.x, self.y)
        # find max
        Qlist = []
        for action in list:
            Qlist.append(self.Q[(sn, action)])

        '''greedily update Q'''
        self.Q[(s, a)] += self.lr * (self.reward + self.gamma*np.max(Qlist) - self.Q[(s, a)])
        # reset reward
        self.reward = 0

        '''return terminal status after action'''
        return ter

    def SARSA(self, ghosts):

        '''check terminal before Pacman moves'''
        '''if terminal, quit'''
        ter = self.game.terminal(self, ghosts)
        if ter:
            self.game.terScore()
            return True

        '''get current state'''
        s = self.getState(ghosts)

        '''get legal actions'''
        aList = self.getLegalActions(self.x, self.y)

        Qlist = []
        for action in aList:
            Qlist.append(self.Q[(s, action)])

        '''e-greedy'''
        pr = np.ones(len(Qlist)) * self.epsilon / len(Qlist)
        pr[np.argmax(Qlist)] += 1 - self.epsilon
        a = np.random.choice(aList, p=pr)

        '''act'''
        self.takeLegalAction(a)
        if self.hasFood(self.x, self.y):
            self.eatFood(self.x, self.y)
        '''check terminal'''
        ter = self.game.terminal(self, ghosts)
        if ter:
            self.game.terScore()
        '''get reward'''
        self.getReward()

        '''get next state'''
        sn = self.getState(ghosts)

        '''______e-greedy Qsn_____'''
        # get legal actions
        list = self.getLegalActions(self.x, self.y)
        # find max
        Qlist = []
        for action in list:
            Qlist.append(self.Q[(sn, action)])
        # e-greedy
        pr = np.ones(len(Qlist)) * self.epsilon / len(Qlist)
        pr[np.argmax(Qlist)] += 1 - self.epsilon
        Qsn = np.random.choice(Qlist, p=pr)

        '''e-greedy update Q'''
        self.Q[(s, a)] += self.lr * (self.reward + self.gamma*Qsn - self.Q[(s, a)])
        # reset reward
        self.reward = 0

        '''return terminal status after action'''
        return ter

    def ApproxQ(self, ghosts):

        # print(self.weight)
        '''check terminal before Pacman moves'''
        '''if terminal, quit'''
        ter =  self.game.terminal(self, ghosts)
        if ter:
            self.game.terScore()
            return True

        '''check scared ghosts'''
        for ghost in ghosts:
            if self.hasOneScaredGhost(self.x, self.y, ghost):
                # when ghost is eaten, score counts and ghost reset
                self.eatGhost()
                ghost.reset()

        # '''get current state'''
        # s = self.getState(ghosts)

        '''get legal actions'''
        aList = self.getLegalActions(self.x, self.y)

        '''___get Q from features and weights___'''
        Qlist = []
        for action in aList:
            fs = self.getFeatures(action, ghosts)
            Qs = self.QFromFeatures(fs)
            Qlist.append(Qs)

        '''e-greedy'''
        pr = np.ones(len(Qlist))*self.epsilon/len(Qlist)
        pr[np.argmax(Qlist)] += 1 - self.epsilon
        a = np.random.choice(aList, p=pr)
        # from a get f(s, a) and then get Q(s, a)
        fs = self.getFeatures(a, ghosts)
        Qs = self.QFromFeatures(fs)


        '''act'''
        self.takeLegalAction(a)
        if self.hasFood(self.x, self.y):
            self.eatFood(self.x, self.y)
        '''eat capsule and scare ghosts'''
        if self.hasCapsule(self.x, self.y):
            self.eatCapsule(self.x, self.y)
            for ghost in ghosts:
                ghost.scared = True
                ghost.scareTimer = ghost.scareTimerAll
        '''check scared ghosts'''
        for ghost in ghosts:
            if self.hasOneScaredGhost(self.x, self.y, ghost):
                # when ghost is eaten, score counts and ghost reset
                self.eatGhost()
                ghost.reset()
        '''check terminal'''
        ter = self.game.terminal(self, ghosts)
        if ter:
            self.game.terScore()
        '''get reward'''
        self.getReward()

        # '''get next state'''
        # sn = self.getState(ghosts)

        '''max Qsn'''
        # get legal actions
        aList = self.getLegalActions(self.x, self.y)
        # find max
        Qlist = []
        for action in aList:
            fsn = self.getFeatures(action, ghosts)
            Qsn = self.QFromFeatures(fsn)
            Qlist.append(Qsn)

        '''greedily update weights'''
        for feature in fs.keys():
            self.weight[feature] += self.game.lr * fs[feature] \
                               * (self.reward + self.game.gamma * np.max(Qlist) - Qs)
        # reset reward
        self.reward = 0

        '''return terminal status after action'''
        return ter

class Run():

    def __init__(self, game, pacman, ghosts):
        self.trainResults = []
        self.testResults = []
        self.results = []
        self.winRate = []
        self.scores = []
        self.game = game
        self.pacman = pacman
        self.ghosts = ghosts

    def train(self):
        # environment display
        self.game.screen.fill(self.game.BLACK)
        self.game.mapShow()

        # time penalty
        self.game.timeScore()

        # agents action
        for ghost in self.ghosts:
            # ghost.move()
            ghost.move()
        # choose mode
        if self.game.mode == "manual":
            epiEnd = self.pacman.manual(self.ghosts)
        elif self.game.mode == "QLearning":
            epiEnd = self.pacman.QLearning(self.ghosts)
        elif self.game.mode == "SARSA":
            epiEnd = self.pacman.SARSA(self.ghosts)
        elif self.game.mode == "ApproxQ":
            epiEnd = self.pacman.ApproxQ(self.ghosts)
        else:
            # default: manual
            epiEnd = self.pacman.manual(self.ghosts)

        # agents display
        self.pacman.show()
        for ghost in self.ghosts:
            ghost.show()

        # score/episode display
        self.game.scoreShow()
        self.game.episodeShow()
        self.game.phaseShow("Training")

        return epiEnd

    def test(self):
        self.pacman.epsilon = 0
        self.pacman.lr = 0

        # environment display
        self.game.screen.fill(self.game.BLACK)
        self.game.mapShow()

        # time penalty
        self.game.timeScore()

        # agents actions
        for ghost in self.ghosts:
            ghost.move()
        # choose mode; manual is default
        if self.game.mode == "manual":
            epiEnd = self.pacman.manual(self.ghosts)
        elif self.game.mode == "QLearning":
            epiEnd = self.pacman.QLearning(self.ghosts)
        elif self.game.mode == "SARSA":
            epiEnd = self.pacman.SARSA(self.ghosts)
        elif self.game.mode == "ApproxQ":
            epiEnd = self.pacman.ApproxQ(self.ghosts)
        else:
            epiEnd = self.pacman.manual(self.ghosts)

        # agents display
        self.pacman.show()
        for ghost in self.ghosts:
            ghost.show()

        # score/episode display
        self.game.scoreShow()
        self.game.episodeShow()
        self.game.phaseShow("Testing")

        return epiEnd

    def final(self, mode):
        print("_________Conclusion___________")
        if mode == "train":
            print(self.trainResults)
            won = self.trainResults.count("won")
            total = self.game.trainEpi
            print("Winrate: "+str(won)+"/"+str(total)+"="+str(won / total * 100)+"%"
                  + ", Average Score: "+str(sum(self.scores)/total))
        else:
            print(self.testResults)
            won = self.testResults.count("won")
            total = self.game.testEpi
            print("Winrate: "+str(won) + "/" + str(total)+"="+str(won / total * 100)+"%"
                  + ", Average Score: "+str(sum(self.scores)/total))

        # print((str)(self.results.count("won")) + "/" + (str)(self.game.trainEpi))
        return self.winRate

    def checkDone(self):
        done = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == KEYDOWN:
                if event.key == K_q:
                    done = True
        return done

    def loopTrain(self):
        while self.game.episode <= self.game.trainEpi:
            done = self.checkDone()
            epiEnd = self.train()
            pygame.time.delay(self.game.trainDelay)

            '''display turned OFF for training'''
            '''until last 10 rounds'''
            # if self.game.trainEpi - self.game.episode < 10:
            #     pygame.display.flip()
            '''display all training(OFF now)'''
            pygame.display.flip()

            if epiEnd:
                if self.game.lost:
                    self.trainResults.append("lost")
                elif self.game.won: self.trainResults.append("won")
                self.winRate.append(self.trainResults.count("won")/self.game.episode)
                self.scores.append(self.game.score)

                self.game.episode += 1
                self.game.reset()
                self.pacman.reset()
                for ghost in self.ghosts:
                    ghost.reset()

            '''quit'''
            if done:
                return done

    def loopTest(self):
        while self.game.episode <= self.game.testEpi:
            done = self.checkDone()
            epiEnd = self.test()
            pygame.time.delay(self.game.testDelay)

            '''display turned OFF for training'''
            '''until last 10 rounds'''
            # if self.game.trainEpi - self.game.episode < 10:
            #     pygame.display.flip()
            '''display all training(OFF now)'''
            pygame.display.flip()

            if epiEnd:
                if self.game.lost: self.testResults.append("lost")
                elif self.game.won: self.testResults.append("won")
                self.winRate.append(self.testResults.count("won") / self.game.episode)
                self.scores.append(self.game.score)

                self.game.episode += 1
                self.game.reset()
                self.pacman.reset()
                for ghost in self.ghosts:
                    ghost.reset()

            '''quit'''
            if done:
                return done

    def plotTwo(self, trainWR, testWR, labels):
        trainL = len(trainWR)
        testL = len(testWR)
        xTrain = np.arange(1, trainL+1)
        xTest = np.arange(1, testL+1)
        graph = []
        gTrain, = plt.plot(xTrain, trainWR, label=labels[0])
        gTest, = plt.plot(xTest, testWR, label=labels[1])
        graph.append(gTrain)
        graph.append(gTest)
        # for arg in args, lab in labels:
        #     g, = plt.plot(arg, label=lab)
        #     graph.append(g)
        plt.title("Winrate against Episode")
        plt.xlabel("Episode")
        plt.ylabel("Winrate")
        plt.legend(graph, labels)
        plt.show()

    def plot(self, WR, label):
        l = len(WR)
        xl = np.arange(1, l+1)
        g, = plt.plot(xl, WR, label=label)
        graph = [g]
        plt.title("Winrate against Episode")
        plt.xlabel("Episode")
        plt.ylabel("Winrate")
        plt.legend(graph, label)
        plt.show()

    def flow(self):
        print("_____________TRAINING_______________")
        done = self.loopTrain()
        if done:
            pygame.quit()
            return
        out = self.final("train")
        trainWR = deepcopy(out)
        trainScores = deepcopy(self.scores)

        self.winRate = []
        self.results = []
        self.scores = []
        self.game.episode = 1
        print("_____________TESTING_______________")
        done = self.loopTest()
        if done:
            pygame.quit()
            return

        testWR = self.final("test")
        testScores = self.scores

        '''Plot WR of training and testing'''
        '''only work if train/test have same episodes'''
        labels = ["train", "test"]
        label = ["Training WinRate"]
        self.plot(trainWR, label)
        self.plotTwo(trainWR, testWR, labels)
        self.plotTwo(trainScores, testScores, labels)

        pygame.quit()

    '''_______________MCTS______________'''

    def initialState(self):

        self.game.reset()
        self.pacman.reset()
        ghost = self.ghosts[0]
        ghost.reset()

        state = self.pacman.getState(self.ghosts)
        return State(state)

    def MCST_Train(self, mode):
        # node = self.loopTrain_MCTS()
        # print(node.ValueAll)
        iniS = self.initialState()
        root = TreeNode(iniS, None, None)
        trainEpi = self.game.trainEpi


        for t in range(trainEpi):

            node = root
            S = iniS.clone()
            visited = [root]
            print("iteration: ", t)

            # while not terminal
            while(S.getReward()==0):

                pac = S.state[0]
                for row in pac:
                    print(row)
                print("________________")

                # select(fully expanded and non terminal)
                while node.children != []:
                    node = node.choose()
                    S.act(node.action)
                    visited.append(node)
                    # print(1)

                # expand
                # else:
                a = np.random.choice(node.untriedMoves)
                S.act(a)
                # add child and traverse down the tree
                node = node.addChild(a, S)
                visited.append(node)
                # print(2)

                    # # rollout
                    # input = S.state[S.player]
                    # grid = S.state[3]
                    # aList = S.getLegalActions(input, grid)
                    # while aList != []:
                    #     S.act(np.random.choice(aList))
                    # #     # print(3)

            # backpropagate
            for node in visited:
                node.update(S.getReward())

                # print(3)

        # print info
        if mode:
            print(root.treeToStr(0))
        else:
            print(root.childrenToStr())

        # return the most visited move
        return sorted(root.children, key=lambda c: c.visited)[-1].move

    # def MCST_Train_outofbound(self, mode):
    #     # node = self.loopTrain_MCTS()
    #     # print(node.ValueAll)
    #     iniS = self.initialState()
    #     root = TreeNode(iniS, None, None)
    #     trainEpi = self.game.trainEpi
    #
    #     for t in range(trainEpi):
    #
    #         node = root
    #         S = iniS.clone()
    #         print("iteration: ", t)
    #
    #         while(S.getReward()==0):
    #
    #             # select(fully expanded and non terminal)
    #             if node.untriedMoves == [] and node.children != []:
    #                 node = node.choose()
    #                 S.act(node.action)
    #                 # print(1)
    #
    #             # expand
    #             elif node.untriedMoves != []: #non terminal
    #                 a = np.random.choice(node.untriedMoves)
    #                 S.act(a)
    #                 # add child and traverse down the tree
    #                 node = node.addChild(a, S)
    #                 # print(2)
    #
    #                 # rollout(non terminal)
    #                 input = S.state[S.player]
    #                 grid = S.state[3]
    #                 aList = S.getLegalActions(input, grid)
    #                 while aList != []:
    #                     S.act(np.random.choice(aList))
    #                     # print(3)
    #
    #         # backpropagate
    #         while node != None:
    #             node.update(S.getReward())
    #             node = node.parent
    #             # print(4)
    #
    #     # print info
    #     if mode:
    #         print(root.treeToStr(0))
    #     else:
    #         print(root.childrenToStr())
    #
    #     # return the most visited move
    #     return sorted(root.children, key=lambda c: c.visited)[-1].move

    # def MCST_Train_BU(self, mode):
    #     # node = self.loopTrain_MCTS()
    #     # print(node.ValueAll)
    #     iniState = self.initialState()
    #     root = TreeNode(iniState, None, None)
    #     trainEpi = self.game.trainEpi
    #
    #     for t in range(trainEpi):
    #
    #         node = root
    #         state = deepcopy(iniState)
    #         print("iteration: ", t)
    #
    #         # select(fully expanded and non terminal)
    #         while node.untriedMoves == [] and node.children != [] and state.getReward() == 0:
    #             node = node.choose()
    #             state.act(node.action)
    #             print(1)
    #
    #         # expand
    #         if node.untriedMoves != [] and state.getReward() == 0: #non terminal
    #             a = np.random.choice(node.untriedMoves)
    #             state.act(a)
    #             # add child and traverse down the tree
    #             node = node.addChild(a, state)
    #             print(2)
    #
    #         # rollout(non terminal)
    #         input = state.state[state.player]
    #         grid = state.state[3]
    #         while state.getLegalActions(input, grid) != []\
    #                 and state.getReward() == 0:
    #             state.act(np.random.choice(state.getLegalActions(input, grid)))
    #             print(3)
    #
    #         # backpropagate
    #         while node != None:
    #             node.update(state.getReward())
    #             node = node.parent
    #             print(4)
    #
    #     # print info
    #     if mode:
    #         print(root.treeToStr(0))
    #     else:
    #         print(root.childrenToStr())
    #
    #     # return the most visited move
    #     return sorted(root.children, key=lambda c: c.visited)[-1].move

class TreeNode:

    def __init__(self, S, a, parent):
        '''state, action, parent'''
        self.S = S
        self.action = a
        self.parent = parent
        self.children = []
        self.wins = 0
        self.visits = 0
        if self.S.player == 0:
            self.untriedMoves = self.S.getLegalActions(self.S.state[0], self.S.state[3])
        else:
            self.untriedMoves = self.S.getLegalActions(self.S.state[1], self.S.state[3])
        self.player = S.player

    def choose(self):
        '''return nextnode based on UCB1/UCT value'''
        chosen = None
        maxValue = -1.0e10    #initialize some very small value
        for child in self.children:
            # uct/ucb: upper confidence bounds for trees; 1e-6 to avoid divide 0
            uct = child.wins/(child.visits + 1e-6)\
                  +np.sqrt(2*np.log(self.visits+1) / (child.visits + 1e-6))
            if uct > maxValue:
                chosen = child
                maxValue = uct
        return chosen

    def addChild(self, action, S):
        # remove action from untriedMoves
        # add new child node for this action
        # return the child node
        newNode = TreeNode(S, action, self)
        self.untriedMoves.remove(action)
        self.children.append(newNode)
        return newNode

    def update(self, reward):
        self.visits += 1
        self.wins += reward

    '''_________Visualize Tree___________'''

    def indentStr(self, indent):
        s = "\n"
        for i in range(1, indent + 1):
            s += "| "
        return s

    def treeToStr(self, indent):
        s = self.indentStr(indent) + str(self)
        for c in self.children:
            s += c.TreeToString(indent + 1)
        return s

    def childrenToStr(self):
        s = ""
        for c in self.children:
            s += str(c) + "\n"
        return s

class State:

    def __init__(self, state):
        # pacman: 0 ; ghost: 1
        # pacman move first(root)
        self.player = 0
        '''pac, ghosts, food, wall'''
        self.state = state

    '''____Read state channel, find next state, build state channel____'''
    def findOnes(self, input):
        '''find the 1s in the input tuple'''
        oneList = []
        xA, yA = len(input[0]), len(input)
        for y in range(yA):
            for x in range(xA):
                if input[y][x] == 1:
                    oneList.append((x, y))
        return oneList

    def nextXY(self, x, y, a):
        if a == 0: return (x-1, y)
        if a == 1: return (x+1, y)
        if a == 2: return (x, y-1)
        if a == 3: return (x, y+1)

    def buildState(self, list, xA, yA):
        matrix = []
        for i in range(yA):
            matrix.append([])
            for j in range(xA):
                if (i, j) in list:
                    matrix[-1].append(1)
                else:
                    matrix[-1].append(0)
            matrix[-1] = tuple(matrix[-1])
        return tuple(matrix)

    '''__________________update state channels_________________'''
    def pacAct(self, input, a):
        xA, yA = len(input[0]), len(input)

        # build new pacman state matrix
        curxy = self.findOnes(input)
        cx, cy = curxy[0]
        nx, ny = self.nextXY(cx, cy, a)
        pacNS = self.buildState([(nx, ny)], xA, yA)
        return pacNS, (nx, ny)

    def foodAct(self, input, sp):
        nx, ny = sp
        xA, yA = len(input[0]), len(input)
        if input[ny][nx] == 1:
            foodList = self.findOnes(input)
            foodList.remove((nx, ny))
            foodNS = self.buildState(foodList, xA, yA)
            return foodNS
        else:
            return input

    def ghoAct(self, input, a):
        xA, yA = len(input[0]), len(input)

        # build new pacman state matrix
        curxy = self.findOnes(input)
        cx, cy = curxy[0]
        nx, ny = self.nextXY(cx, cy, a)
        # nextxy = []
        # for i in range(len(curxy)):
        #     cx, cy = curxy[i]
        #     nxy = self.nextXY(cx, cy, a)
        #     nextxy.append(nxy)
        ghostNS = self.buildState([(nx, ny)], xA, yA)
        # ghostNS = self.buildState(nextxy, xA, yA)
        return ghostNS

    '''_______________others_________________'''
    def act(self, action):
        '''act and update state'''

        # pacman
        if self.player == 0:
            statePac, sp = self.pacAct(self.state[0], action)
            stateFood = self.foodAct(self.state[2], sp)
            self.state = (statePac, self.state[1], stateFood, self.state[3])
        # ghost
        else:
            stateGho = self.ghoAct(self.state[1], action)
            self.state = (self.state[0], stateGho, self.state[2], self.state[3])

        self.player = 1 - self.player

    def clone(self):
        newS = State(self.state)
        return newS

    def getLegalActions(self, input, grid):
        xA, yA = len(input[0]), len(input)
        curxy = self.findOnes(input)
        cx, cy = curxy[0]
        aList = []
        if cx > 0 and grid[cy][cx-1] == 0: aList.append(0)
        if cx < xA-1 and grid[cy][cx+1] == 0: aList.append(1)
        if cy > 0 and grid[cy-1][cx] == 0: aList.append(2)
        if cy < yA-1 and grid[cy+1][cx] == 0: aList.append(3)
        # print(aList)
        return aList

    def getReward(self):
        state = self.state
        # for matrix in state:
        #     for row in matrix:
        #         print(row)
        #     print("\n")
        # lost
        pac = self.findOnes(state[0])[0]
        gho = self.findOnes(state[1])[0]
        if pac == gho:
            return -1
        # win
        foodList = self.findOnes(state[2])[0]
        if foodList == []:
            return 1
        # not terminal
        return 0