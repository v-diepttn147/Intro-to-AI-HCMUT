# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first. Using graph search algorithm

    Input: problem
    Output: List of actions that reaches the goal

    """

    "*** YOUR CODE HERE ***"

    # init frontier including the start state:
    frontier = util.Stack()
    # a frontier item include current state and movements to get there from start state
    frontier.push((problem.getStartState(), []))

    # init empty movements list and explored nodes list
    moves = []
    explored = []

    while not frontier.isEmpty():
        (state, path) = frontier.pop()
        # check if the state is goal
        if problem.isGoalState(state):
            moves = path
            break
        
        # check if the state is not visited
        elif state not in explored:
            explored.append(state)
            for child in problem.getSuccessors(state):
                # add the new move
                newPath = path + [child[1]]
                # create new state
                newState = (child[0], newPath)
                frontier.push(newState)

    return moves


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""

    "*** YOUR CODE HERE ***"
    
    # init frontier including the start state:
    frontier = util.Queue()
    # a frontier item include current state and movements to get there from start state
    frontier.push((problem.getStartState(), []))
    # init empty movements list and explored nodes list
    moves = []
    explored = []
    while not frontier.isEmpty():
        (state, path) = frontier.pop()
        # check if the state is goal
        if problem.isGoalState(state):
            moves = path
            break 
            
        
        # check if the state is not visited
        elif state not in explored:
            explored.append(state)
            for child in problem.getSuccessors(state):
                # add the new move
                newPath = path + [child[1]]
                # create new state
                newState = (child[0], newPath)
                frontier.push(newState)

    return moves
    

def uniformCostSearch(problem):
    """Search the node of least total cost first."""

    "*** YOUR CODE HERE ***"

    # init frontier including the start state:
    frontier = util.PriorityQueue()
    # a frontier item include current state and movements to get there from start state
    startNode = problem.getStartState()
    # print(startNode) - (34, 16)
    # print(problem.getSuccessors(problem.getStartState()))
    # [((34, 15), 'South', 5.820766091346741e-11), ((33, 16), 'West', 1.1641532182693481e-10)]
    frontier.push(startNode, 0)

    # init empty movements list and explored nodes list
    moves = []
    explored = []
    # movement dictionary to each node
    move_dict = {str(startNode): []}

    while not frontier.isEmpty():
        state = frontier.pop()
        path = move_dict[str(state)]
        # check if the state is goal
        if problem.isGoalState(state):
            moves = path
            break
        
        # state is explored
        explored.append(state)

        for child in problem.getSuccessors(state):
            # path to child node
            newPath = path + [child[1]]
            newState = child[0]
            # if the child is not explored yet
            if child[0] not in explored:
                # add / update frontier
                frontier.update(newState, problem.getCostOfActions(newPath))
                # add / update path movement dictionary
                if str(newState) in move_dict.keys() and problem.getCostOfActions(move_dict[str(newState)]) > problem.getCostOfActions(newPath):
                    move_dict[str(newState)] = newPath
                elif str(newState) not in move_dict.keys():
                    move_dict[str(newState)] = newPath

    return moves

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    "*** YOUR CODE HERE ***"

    # init frontier including the start state:
    frontier = util.PriorityQueue()
    # a frontier item include current state and movements to get there from start state
    startNode = problem.getStartState()
    frontier.push(startNode, 0)

    # init empty movements list and explored nodes list
    moves = []
    explored = []
    # movement dictionary to each node
    move_dict = {str(startNode): []}

    while not frontier.isEmpty():
        state = frontier.pop()
        path = move_dict[str(state)]
        # check if the state is goal
        if problem.isGoalState(state):
            moves = path
            break
        
        # state is explored
        explored.append(state)

        for child in problem.getSuccessors(state):
            # path to child node
            newPath = path + [child[1]]
            newState = child[0]
            # if the child is not explored yet
            if child[0] not in explored:
                # add / update frontier
                frontier.update(newState, problem.getCostOfActions(newPath)+ heuristic(newState, problem))
                # add / update path movement dictionary
                if str(newState) in move_dict.keys() and problem.getCostOfActions(move_dict[str(newState)]) > problem.getCostOfActions(newPath):
                    move_dict[str(newState)] = newPath
                elif str(newState) not in move_dict.keys():
                    move_dict[str(newState)] = newPath

    return moves


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
