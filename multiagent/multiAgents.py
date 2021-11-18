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


import random
import util

from game import Agent


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
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        newFoodList = newFood.asList()
        newFoodDistances = []
        for food in newFoodList:
            newFoodDistances.append(util.manhattanDistance(newPos, food))

        newGhostDistances = []
        for ghostState in newGhostStates:
            newGhostDistances.append(util.manhattanDistance(newPos, ghostState.getPosition()))

        f1 = min(newGhostDistances)
        if f1 == 0:
            f1 += 1
        f2 = 1
        if len(newFoodDistances) != 0:
            f2 = min(newFoodDistances)
        f3 = len(newFoodList)
        f4 = 0
        if len(newScaredTimes) != 0:
            f4 = min(newScaredTimes)

        return -2/f1 + 1/(2*f2) - f3 + f4

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

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
        return self.value(gameState, 0, self.index)[1]

    def value(self, gameState, depth, index):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), None

        amount_of_agents = gameState.getNumAgents()
        # checking if is at max depth after expanding the last index
        if depth == self.depth and index == 0:
            return self.evaluationFunction(gameState), None

        if index == 0:
            return self.max_value(gameState, depth, index)

        return self.min_value(gameState, depth, index)

    def max_value(self, gameState, depth, index):
        # max_value call means next depth
        depth += 1
        legalActions = gameState.getLegalActions(index)

        if len(legalActions) == 0:
            return self.evaluationFunction(gameState), None

        new_index = index + 1
        v = float('-inf')
        v_action = None
        for legalAction in legalActions:
            successor = gameState.generateSuccessor(index, legalAction)
            new_value = self.value(successor, depth, new_index)[0]
            if new_value > v:
                v = new_value
                v_action = legalAction
        return v, v_action

    def min_value(self, gameState, depth, index):
        legalActions = gameState.getLegalActions(index)

        if len(legalActions) == 0:
            return self.evaluationFunction(gameState), None

        amount_of_agents = gameState.getNumAgents()

        new_index = None
        assert index <= amount_of_agents
        if index == amount_of_agents - 1:
            new_index = 0
        elif index < amount_of_agents:
            new_index = index + 1
        assert new_index is not None

        v = float('+inf')
        v_action = None
        for legalAction in legalActions:
            successor = gameState.generateSuccessor(index, legalAction)
            new_value = self.value(successor, depth, new_index)[0]
            if new_value < v:
                v = new_value
                v_action = legalAction
        return v, v_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.value(gameState, 0, self.index, float('-inf'), float('+inf'))[1]

    def value(self, gameState, depth, index, alpha, beta):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), None

        amount_of_agents = gameState.getNumAgents()
        # checking if is at max depth after expanding the last index
        if depth == self.depth and index == 0:
            return self.evaluationFunction(gameState), None

        if index == 0:
            return self.max_value(gameState, depth, index, alpha, beta)

        return self.min_value(gameState, depth, index, alpha, beta)

    def max_value(self, gameState, depth, index, alpha, beta):
        # max_value call means next depth
        depth += 1
        legalActions = gameState.getLegalActions(index)
        if len(legalActions) == 0:
            return self.evaluationFunction(gameState), None

        new_index = index + 1
        v = float('-inf')
        v_action = None
        for legalAction in legalActions:
            successor = gameState.generateSuccessor(index, legalAction)
            new_value = self.value(successor, depth, new_index, alpha, beta)[0]
            if new_value > v:
                v = new_value
                v_action = legalAction
            if v > beta:
                return v, v_action
            alpha = max(alpha, v)
        return v, v_action

    def min_value(self, gameState, depth, index, alpha, beta):
        legalActions = gameState.getLegalActions(index)
        if len(legalActions) == 0:
            return self.evaluationFunction(gameState), None

        amount_of_agents = gameState.getNumAgents()

        new_index = None
        assert index <= amount_of_agents
        if index == amount_of_agents - 1:
            new_index = 0
        elif index < amount_of_agents:
            new_index = index + 1
        assert new_index is not None

        v = float('+inf')
        v_action = None
        for legalAction in legalActions:
            successor = gameState.generateSuccessor(index, legalAction)
            new_value = self.value(successor, depth, new_index, alpha, beta)[0]
            if new_value < v:
                v = new_value
                v_action = legalAction
            if v < alpha:
                return v, v_action
            beta = min(beta, v)
        return v, v_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.value(gameState, 0, self.index)[1]

    def value(self, gameState, depth, index):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), None

        amount_of_agents = gameState.getNumAgents()
        # checking if is at max depth after expanding the last index
        if depth == self.depth and index == 0:
            return self.evaluationFunction(gameState), None

        if index == 0:
            return self.max_value(gameState, depth, index)

        return self.exp_value(gameState, depth, index)

    def max_value(self, gameState, depth, index):

        # max_value call means next depth
        depth += 1
        legalActions = gameState.getLegalActions(index)

        if len(legalActions) == 0:
            return self.evaluationFunction(gameState), None

        new_index = index + 1
        v = float('-inf')
        v_action = None
        for legalAction in legalActions:
            successor = gameState.generateSuccessor(index, legalAction)
            new_value = self.value(successor, depth, new_index)[0]
            if new_value > v:
                v = new_value
                v_action = legalAction
        return v, v_action

    def exp_value(self, gameState, depth, index):
        legalActions = gameState.getLegalActions(index)

        if len(legalActions) == 0:
            return self.evaluationFunction(gameState), None


        amount_of_agents = gameState.getNumAgents()

        new_index = None
        assert index <= amount_of_agents
        if index == amount_of_agents - 1:
            new_index = 0
        elif index < amount_of_agents:
            new_index = index + 1
        assert new_index is not None

        v = 0
        v_action = None
        p = 1 / len(legalActions)
        for legalAction in legalActions:
            successor = gameState.generateSuccessor(index, legalAction)
            v += p * self.value(successor, depth, new_index)[0]
        return v, v_action

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
