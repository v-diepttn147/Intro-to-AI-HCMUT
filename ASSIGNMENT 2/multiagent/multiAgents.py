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


from util import manhattanDistance
from game import Directions
import random, util

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
        
        # get positions of ghosts at new state
        newGhostPositions = [ghost.getPosition() for ghost in newGhostStates if ghost.scaredTimer == 0]
        scared = min(newScaredTimes) > 0 # if there is at least one scared ghost
        # print(newGhostStates[0].scaredTimer)

        # if next position is ghost and not scared: return lowest value
        if (newPos in newGhostPositions) and not scared:
            return -1.0

        # get list of food in current state
        newFood = currentGameState.getFood().asList()

        # if next position is food: return highest value
        if newPos in newFood:
            return 1.0

        # get closest list to food and ghost
        ghostDistanceList = [manhattanDistance(newPos, ghost) for ghost in newGhostPositions]
        foodDistanceList = [manhattanDistance(newPos, food) for food in newFood]
        
        # get the nearest food dot
        if len(foodDistanceList) > 0: closestFoodDistance = min(foodDistanceList)
        else: closestFoodDistance = -1.0

        # get the nearest ghost
        if len(ghostDistanceList) > 0: closestGhostDistance = min(ghostDistanceList)
        else: closestGhostDistance = -1.0
        # print(closestGhostDistance)
        return 1.0 / closestFoodDistance - 1.0 / closestGhostDistance
        

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

        ghostsIndex = [i for i in range(1, gameState.getNumAgents())]
        infinity = 1e100
        
        # determine if the state is goal or at the leaf
        def isStop(state, depth):
            return state.isWin() or state.isLose() or depth == self.depth

        # minimize function for ghosts
        def minimize(state, depth, ghostIndex):
            if isStop(state, depth):
                return self.evaluationFunction(state)
            value = infinity
            for action in state.getLegalActions(ghostIndex):
                if ghostIndex == ghostsIndex[-1]:
                    value = min(value, maximize(state.generateSuccessor(ghostIndex, action), depth + 1))
                else:
                    value = min(value, minimize(state.generateSuccessor(ghostIndex, action), depth, ghostIndex + 1))
            return value

        # maximize function for pacman
        def maximize(state, depth, pacmanIndex = 0):
            if isStop(state, depth):
                return self.evaluationFunction(state)
            value = -infinity
            for action in state.getLegalActions(pacmanIndex):
                value = max(value, minimize(state.generateSuccessor(pacmanIndex, action), depth, 1))
            return value
        
        # call the minimize function first
        result = [(action, minimize(gameState.generateSuccessor(0, action), 0, 1)) for action in gameState.getLegalActions(0)]
        result.sort(key=lambda k: k[1])

        return result[-1][0]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        ghostsIndex = [i for i in range(1, gameState.getNumAgents())]
        infinity = 1e100
        
        # determine if the state is goal or at the leaf
        def isStop(state, depth):
            return state.isWin() or state.isLose() or depth == self.depth

        # minimize function for ghosts
        def minimize(state, depth, alpha, beta, ghostIndex):
            if isStop(state, depth):
                return self.evaluationFunction(state)
            value = infinity
            for action in state.getLegalActions(ghostIndex):
                if ghostIndex == ghostsIndex[-1]:
                    value = min(value, maximize(state.generateSuccessor(ghostIndex, action), depth + 1, alpha, beta))
                    # beta = min(beta, value)
                    # if alpha >= beta: break
                else:
                    value = min(value, minimize(state.generateSuccessor(ghostIndex, action), depth, alpha, beta, ghostIndex + 1))
                if value < alpha:
                    return value
                beta = min(beta, value)
                # if alpha >= beta: break
            return value

        # maximize function for pacman
        def maximize(state, depth, alpha, beta, pacmanIndex = 0):
            if isStop(state, depth):
                return self.evaluationFunction(state)
            value = -infinity
            for action in state.getLegalActions(pacmanIndex):
                value = max(value, minimize(state.generateSuccessor(pacmanIndex, action), depth, alpha, beta, 1))
                # alpha = max(alpha, value)
                # if alpha >= beta: break
                if value > beta:
                    return value
                alpha = max(alpha, value)
            return value
        
        alpha = -infinity
        beta = infinity
        value = -infinity
        return_action = None

        # run the alpha beta pruning here
        for action in gameState.getLegalActions(0):
            tmp_value = minimize(gameState.generateSuccessor(0, action), 0, alpha, beta, 1)
            if value < tmp_value:
                value = tmp_value
                return_action = action
            if value > beta:
                return value
            alpha = max(alpha, tmp_value)
        return return_action


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

        ghostsIndex = [i for i in range(1, gameState.getNumAgents())]
        infinity = 1e100

        # determine if the state is goal or at the leaf
        def isStop(state, depth):
            return state.isWin() or state.isLose() or depth == self.depth

        # expect function for ghosts
        def expect(state, depth, ghostIndex):
            if isStop(state, depth):
                return self.evaluationFunction(state)
            value = 0
            # the probability that a ghost will choose a specific direction
            probability  = 1 / len(state.getLegalActions(ghostIndex))

            for action in state.getLegalActions(ghostIndex):
                if ghostIndex == ghostsIndex[-1]:
                    value += probability * maximize(state.generateSuccessor(ghostIndex, action), depth + 1)
                else:
                    value += probability * expect(state.generateSuccessor(ghostIndex, action), depth, ghostIndex + 1)
            return value

        # maximize function for pacman
        def maximize(state, depth, pacmanIndex = 0):
            if isStop(state, depth):
                return self.evaluationFunction(state)
            value = -infinity
            for action in state.getLegalActions(pacmanIndex):
                value = max(value, expect(state.generateSuccessor(pacmanIndex, action), depth, 1))
            return value
        
        # call the expect function first
        result = [(action, expect(gameState.generateSuccessor(0, action), 0, 1)) for action in gameState.getLegalActions(0)]
        result.sort(key=lambda k: k[1])

        return result[-1][0]



def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Features:  
                    Closest ghost
                    Furthest ghost
                    Number of capsules
                    Closest Food dot

    """
    "*** YOUR CODE HERE ***"

    
    newPosition = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood().asList()
    newGhostStates = currentGameState.getGhostStates()
    ghostPositions = [ghost.getPosition() for ghost in newGhostStates]

    # if pacman loses or new position is ghost
    if currentGameState.isLose() or (newPosition in ghostPositions):
        return float('-inf')
    
    # get closest list to food and ghost
    ghostDistanceList = [manhattanDistance(newPosition, ghost) for ghost in ghostPositions]
    foodDistanceList = [manhattanDistance(newPosition, food) for food in newFood]
    
    # get the nearest food dot
    if len(foodDistanceList) > 0: 
        closestFoodDistance = min(foodDistanceList)
    else: 
        closestFoodDistance = 0

    # get the nearest and furthest ghost
    if len(ghostDistanceList) > 0: 
        closestGhostDistance = min(ghostDistanceList)
        furthestGhostDistance = max(ghostDistanceList)
    else: 
        closestGhostDistance = 0

    # init evaluation score
    score = 0

    # inspect the closest ghost
    if closestGhostDistance < 3: 
        score -= 100
    if closestGhostDistance < 2: 
        score -= 1000
    if closestGhostDistance < 1: 
        return float('-inf')

    # inspect the number of capsules
    if len(currentGameState.getCapsules()) < 2: 
        score += 32

    # if there is no ghost or no food
    if len(foodDistanceList) == 0 or len(ghostDistanceList) == 0:
        score += scoreEvaluationFunction(currentGameState) + 13
    # the final evaluation score
    else:
        score += (scoreEvaluationFunction(currentGameState) + 10.0/closestFoodDistance + 1.0/closestGhostDistance + 1.0/furthestGhostDistance)

    return score
    
# Abbreviation
better = betterEvaluationFunction
