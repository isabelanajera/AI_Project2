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
        v = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == v]
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

        # to improve this, we will be using what you give us, like the remaining food and
        # the pacman position after moving, and scared ghosts


        "*** YOUR CODE HERE ***"

        # start off with 0
        score = 0
        
        # we want to encourage winning so if the sucesssor state is winning then
        # return super high value
        if successorGameState.isWin():
            return 999999

        # evaluating food portion
        # hint to use newFood asList function
        foodList = newFood.asList()
        foodDistance = [0]
        # calculate manhattan distance to the available food
        for pos in foodList:
            foodDistance.append(manhattanDistance(newPos,pos))
            
        # number of available food in the game
        availableFood = len(foodList)
        # number of remaining food in the current state
        remainingFood = len(currentGameState.getFood().asList())
        # number of how many capsules there are
        capsules = len(successorGameState.getCapsules())
       
        # if there is available food, subtract from score    
        score -= 10 * availableFood

        # add score if pacman eats capsule to encourage
        if newPos in currentGameState.getCapsules():
            score += 150 * capsules

        # add score if there is less available food 
        if availableFood < remainingFood:
            score += 200


        # evaluating ghosts portion
        # we will get the pos of ghosts in the successor state and then calculate 
        # manhattan distance of the player and the ghosts
        ghostPositions = [ghost.getPosition() for ghost in newGhostStates]
        ghostDist = [manhattanDistance(newPos, pos) for pos in ghostPositions]

        # now we find the position of ghost in current state and calculate that
        # manhattan distance
        currentGhostPositions = [ghost.getPosition() for ghost in currentGameState.getGhostStates()]
        currentGhostDist = [manhattanDistance(newPos, pos) for pos in currentGhostPositions]

        # if ghosts are scared, then less distance is better else if ghosts are not 
        # scared, deduct score to encourage adding distance
        totalScaredTimes = sum(newScaredTimes)
        if totalScaredTimes > 0 :
            if min(currentGhostDist) < min(ghostDist):
                score += 200
            else:
                score -=100
        else:
            if min(currentGhostDist) < min(ghostDist):
                score -= 100
            else:
                score += 200
        
        # add the difference in game score between the successor state and the current state
        # to evaluate if the move is improved or was worse
        score += successorGameState.getScore() - currentGameState.getScore()
        
        # deduct points if stopping
        if action == Directions.STOP:
            score -= 10

        return score

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
            v = float('-inf') # Worst score for max
            bestAction = None
            actions = state.getLegalActions(0)
            
            if state.isWin() or state.isLose(): # No action left
                return self.evaluationFunction(state)
            
            for action in actions:
                score = minimax(state.generateSuccessor(0, action), depth, 1)  # Go to first ghost
                if score > v:
                    v = score
                    bestAction = action
        
            # (Root) return action else score
            return bestAction if depth == 0 else v
        
        # Minimizer function for ghost action with worst score
        def getMin(state, depth, index):
            v = float('inf') # Worst possible score
            actions = state.getLegalActions(index)
            
            if state.isWin() or state.isLose(): # No actions left
                return self.evaluationFunction(state)

            for action in actions:
                if index + 1 == state.getNumAgents():  # If last ghost
                    score = minimax(state.generateSuccessor(index, action), depth + 1, 0) # Next depth
                else:  # Move to the next ghost
                    score = minimax(state.generateSuccessor(index, action), depth, index + 1)
                # Lowest score for pacman    
                v = min(v, score)

            return v


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
            v = float('-inf') # Worst score for max
            bestAction = None
            actions = state.getLegalActions(0)
            
            if state.isWin() or state.isLose(): # No action left
                return self.evaluationFunction(state)
            
            for action in actions:
                score = alphaBeta(state.generateSuccessor(0, action), depth, 1, alpha, beta)  # Go to first ghost
                if score > v:
                    v = score
                    bestAction = action
            
                alpha = max(alpha, v) # Alpha

                if v > beta: # Pruning
                    return v
        
            # (Root) return action else score
            return bestAction if depth == 0 else v
        
        # Minimizer function for ghost action with worst score
        def getMin(state, depth, index, alpha, beta):
            v = float('inf') # Worst possible score
            actions = state.getLegalActions(index)
            
            if state.isWin() or state.isLose(): # No actions left
                return self.evaluationFunction(state)

            for action in actions:
                if index + 1 == state.getNumAgents():  # If last ghost
                    score = alphaBeta(state.generateSuccessor(index, action), depth + 1, 0, alpha, beta) # Next depth
                else:  # Move to the next ghost
                    score = alphaBeta(state.generateSuccessor(index, action), depth, index + 1, alpha, beta)
                # Lowest score for pacman    
                v = min(v, score)
                beta = min(beta, v) # Update beta

                if v < alpha: # Pruning
                    return v

            return v

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
        def expectimax(state, depth, index):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            
            # Recursively call pacman's and ghost's actions
            if index ==0: # Pacman's index
                return getMax(state, depth)
            else:
                return getAvg(state, depth, index)
        # Maximizer funtion for pacman action with highest score          
        def getMax(state, depth):
            v = float('-inf') # Worst score for max
            bestAction = None
            actions = state.getLegalActions(0)
            
            if state.isWin() or state.isLose(): # No action left
                return self.evaluationFunction(state)
            
            for action in actions:
                score = getAvg(state.generateSuccessor(0, action), depth, 1)  # Go to first ghost
                if score > v:
                    v = score
                    bestAction = action
        
            # (Root) return action else score
            return bestAction if depth == 0 else v

        def getAvg(state, depth, index):
            avg = 0
            actions = state.getLegalActions(index)
            
            if state.isWin() or state.isLose(): # No actions left
                return self.evaluationFunction(state)

            for action in actions:
                if index + 1 == state.getNumAgents():  # If last ghost
                    score = expectimax(state.generateSuccessor(index, action), depth + 1, 0) # Next depth
                else:  # Move to the next ghost
                    score = expectimax(state.generateSuccessor(index, action), depth, index + 1)

                avg += score # Sum of all scores
            return avg / len(actions) # The average for expectimax
        
        return expectimax(gameState, 0, 0)

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    This evaluation function takes into account different environmental and other factors to incentivize
    the pacman to choose the best options. The better options, are the ones that gives us high scores.
    We evaluate things like food, ghosts (if they are scared or not) and capsules.

    With food, we calculate the distance to the food, and with the "better" evaluation function, this time 
    we calculate which food is closer, so that we add more points to the score if it goes after closer food first,
    which is overall optimally better and more efficient. 

    With ghosts, we calculate the distance and also if they are scared or not. if they are scared then we 
    subtract the distance with the capsules so that we can incentivize the pacman to go towards them and if they 
    are not scared then add so that they stay away,
    
    """
    "*** YOUR CODE HERE ***"
    # get same info that we got before but from the current game
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    # initialize score
    score = 0

    # evaluating food portion
    # hint to use newFood asList function
    foodList = newFood.asList()
    foodDistance = [0]
    # calculate manhattan distance to the available food
    for pos in foodList:
        foodDistance.append(manhattanDistance(newPos,pos))

    ## CHANGES 
    # number of not available food
    noFood = len(newFood.asList(False))   

    # find closer food to incentivize getting closer food rather than food far away
    closerFood = 0
    foodDistances = sum(foodDistance)
    if foodDistances > 0:
        closerFood = 1.0 / foodDistances
    score += currentGameState.getScore() + closerFood + noFood

    # evaluating ghosts portion
    # we will get the pos of ghosts in the current state and then calculate 
    # manhattan distance of the player and the ghosts
    ghostPositions = [ghost.getPosition() for ghost in newGhostStates]
    ghostDist = [manhattanDistance(newPos, pos) for pos in ghostPositions]
    totalScaredTimes = sum(newScaredTimes)
    totalGhostDist = sum(ghostDist)
    
    capsules = len(currentGameState.getCapsules())

    # if ghost is scared
    if totalScaredTimes > 0:
        # subtract number of distance from ghost from score to get closer to scared ghost
        score += totalScaredTimes + (-1 * capsules) + (-1 * totalGhostDist)
    # if ghost is not scared then add distance and add capsules 
    else :
        score += totalGhostDist + capsules
    return score

   

# Abbreviation
better = betterEvaluationFunction
