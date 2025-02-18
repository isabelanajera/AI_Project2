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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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

        def minimax(state, depth, index):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            
            # Recursively call pacman's and ghost's actions
            if index ==0: # Pacman's index
                return getMax(state, depth)
            else:
                return getMin(state, depth, index)
        # Maximizer funtion for pacman action with highest score          
        def getMax(state, depth):
            bestScore = float('-inf') # Worst score for max
            bestAction = None
            actions = state.getLegalActions(0)
            
            if state.isWin() or state.isLose(): # No action left
                return self.evaluationFunction(state)
            
            for action in actions:
                score = minimax(state.generateSuccessor(0, action), depth, 1)  # Go to first ghost
                if score > bestScore:
                    bestScore = score
                    bestAction = action
        
            # (Root) return action else score
            return bestAction if depth == 0 else bestScore
        
        # Minimizer function for ghost action with worst score
        def getMin(state, depth, index):
            bestScore = float('inf') # Worst possible score
            actions = state.getLegalActions(index)
            
            if state.isWin() or state.isLose(): # No actions left
                return self.evaluationFunction(state)

            for action in actions:
                successor = state.generateSuccessor(index, action)
                if index + 1 == state.getNumAgents():  # If last ghost
                    score = minimax(successor, depth + 1, 0) # Next depth
                else:  # Move to the next ghost
                    score = minimax(successor, depth, index + 1)
                # Lowest score for pacman    
                bestScore = min(bestScore, score)

            return bestScore


        return minimax(gameState, 0, 0)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def alphaBeta(state, depth, index, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            
            # Recursively call pacman's and ghost's actions
            if index ==0:
                return getMax(state, depth, alpha, beta)
            else:
                return getMin(state, depth, index, alpha, beta)
        # Maximizer funtion for pacman action with highest score          
        def getMax(state, depth, alpha, beta):
            bestScore = float('-inf') # Worst score for max
            bestAction = None
            actions = state.getLegalActions(0)
            
            if state.isWin() or state.isLose(): # No action left
                return self.evaluationFunction(state)
            
            for action in actions:
                score = alphaBeta(state.generateSuccessor(0, action), depth, 1, alpha, beta)  # Go to first ghost
                if score > bestScore:
                    bestScore = score
                    bestAction = action
            
                alpha = max(alpha, bestScore) # Alpha

                if bestScore > beta: # Pruning
                    return bestScore
        
            # (Root) return action else score
            return bestAction if depth == 0 else bestScore
        
        # Minimizer function for ghost action with worst score
        def getMin(state, depth, index, alpha, beta):
            bestScore = float('inf') # Worst possible score
            actions = state.getLegalActions(index)
            
            if state.isWin() or state.isLose(): # No actions left
                return self.evaluationFunction(state)

            for action in actions:
                successor = state.generateSuccessor(index, action)
                if index + 1 == state.getNumAgents():  # If last ghost
                    score = alphaBeta(successor, depth + 1, 0, alpha, beta) # Next depth
                else:  # Move to the next ghost
                    score = alphaBeta(successor, depth, index + 1, alpha, beta)
                # Lowest score for pacman    
                bestScore = min(bestScore, score)
                beta = min(beta, bestScore) # Update beta

                if bestScore < alpha: # Pruning
                    return bestScore

            return bestScore

       # Retruns best move at depth 0
        return getMax(gameState, 0, float('-inf'), float('inf'))

    
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
