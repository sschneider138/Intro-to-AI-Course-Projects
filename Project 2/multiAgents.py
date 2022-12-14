# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from cmath import inf
import random

import util
from game import Agent, Directions
from util import manhattanDistance


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        distance = []
        foodList = currentGameState.getFood().asList()
        pacmanPosition = list(successorGameState.getPacmanPosition())

        if action.lower() == "stop":
            return -1 * inf

        for ghost in newGhostStates:
            if ghost.getPosition() == tuple(pacmanPosition) and ghost.scaredTimer == 0:
                return -1 * inf

        for food in foodList:
            x = -1 * abs(food[0] - pacmanPosition[0])
            y = -1 * abs(food[1] - pacmanPosition[1])
            distance.append(x + y)

        return max(distance)


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def miniMax(self, gameState, depth, agent):
        value = [None, None]
        if depth == 0 or gameState.isWin() or gameState.isLose():
            value[0], value[1] = self.evaluationFunction(gameState), None
            return value
        
        if agent == gameState.getNumAgents() - 1:
            depth -= 1
            newAgent = 0
        else:
            newAgent = agent + 1

        allowableActions = gameState.getLegalActions(agent)
        for action in allowableActions:
            next = gameState.generateSuccessor(agent, action)
            nextValue = self.miniMax(next, depth, newAgent)
            if not value[0] and not value[1]:
                value[0] = nextValue[0]
                value[1] = action
            else:
                previousValue = value[0]
                if agent == 0 and nextValue[0] > previousValue:
                    value[0] = nextValue[0]
                    value[1] = action
                elif agent != 0 and nextValue[0] < previousValue:
                    value[0] = nextValue[0]
                    value[1] = action 
        return value

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        value = self.miniMax(gameState, self.depth, 0)
        return value[1]
            



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def max_(self, state, depth, alpha, beta):
        if state.isWin() or state.isLose() or depth==self.depth:
                return self.evaluationFunction(state)
        temp_value=float("-inf")
        for action in state.getLegalActions(0):
            temp_value=max(temp_value,self.min_(state.generateSuccessor(0,action),depth,1,alpha,beta))
            if temp_value>beta:
                return temp_value
            alpha=max(alpha,temp_value)
        return temp_value

    def min_(self, state, depth, agent, alpha, beta):
        if state.isWin() or state.isLose() or depth==self.depth:
                return self.evaluationFunction(state)
        temp_value=float("inf")
        for action in state.getLegalActions(agent):
            if agent==state.getNumAgents()-1:
                temp_value=min(temp_value,self.max_(state.generateSuccessor(agent,action),depth+1,alpha,beta))
            else:
                temp_value=min(temp_value,self.min_(state.generateSuccessor(agent,action),depth,agent+1,alpha,beta))
            if temp_value<alpha:
                return temp_value
            beta=min(beta,temp_value)
        return temp_value

    def getAction(self, gameState):
    #     """
    #     Returns the minimax action using self.depth and self.evaluationFunction
    #     """
    #     "*** YOUR CODE HERE ***"
        temp_val=float("-inf")
        alpha=float("-inf")
        beta=float("inf")
        dir_  = None
        futureActions = gameState.getLegalActions(0)
        for action in futureActions:
            next_state = gameState.generateSuccessor(0, action)
            min_value = self.min_(next_state, 0, 1, alpha, beta)
            if min_value> temp_val:
                temp_val = min_value
                dir_ = action
                alpha = max(min_value, alpha)
        
        return dir_
        # util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def expectiMax(self, gameState, depth, agent):
        value = [None, None]
        if depth == 0 or gameState.isWin() or gameState.isLose():
            value[0], value[1] = self.evaluationFunction(gameState), None
            return value
        
        if agent == gameState.getNumAgents() - 1:
            depth -= 1
            newAgent = 0
        else:
            newAgent = agent + 1
            
        allowableActions = gameState.getLegalActions(agent)
        nextActionsList = []
        for action in allowableActions:
            next = gameState.generateSuccessor(agent, action)
            nextValue = self.expectiMax(next, depth, newAgent)
            if not value[0] and not value[1]:
                if agent == 0:
                    value[0] = nextValue[0]
                    value[1] = action
                else:
                    nextActionsList.append(nextValue[0])
            else:
                previousValue = value[0]
                if agent == 0 and nextValue[0] > previousValue:
                    value[0] = nextValue[0]
                    value[1] = action
                elif agent != 0 and nextValue[0] < previousValue:
                    value[0] = nextValue[0]
                    value[1] = action 
        
        if agent != 0:
            value[0], value[1] = sum(nextActionsList) / len(nextActionsList), None
            return value
        
        return value

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        value = self.expectiMax(gameState, self.depth, 0)
        return value[1]


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    "*** YOUR CODE HERE ***"
    ghost_s = []
    distance_food = []
    distance_ghost = []
    for ghost in newGhostStates:
        ghost_s += [ghost.getPosition()]
    if newPos in ghost_s and 0 in newScaredTimes:
        return -1
    if newPos in currentGameState.getFood().asList():
        return 1
    for i in range(len(newFood.asList())):
        distance_food.append(manhattanDistance(newFood.asList()[i], newPos))
    for j in range(len(ghost_s)):
        distance_ghost.append(manhattanDistance(ghost_s[j], newPos))
    score=0
    if len(currentGameState.getCapsules()) < 2:
        score+=100
    min_food=0 if not distance_food else 1/min(distance_food)
    min_ghost=0 if not distance_food else 1/min(distance_ghost)
    score = score + min_food*5+3*min_ghost+currentGameState.getScore()
    return score

# Abbreviation
better = betterEvaluationFunction
