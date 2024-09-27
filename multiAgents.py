# # multiAgents.py
# --------------

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
        "*** YOUR CODE HERE ***"
        
        newScared = [ghState.scaredTimer for ghState in newGhostStates]
        f_List = newFood.asList()
        gh_score =0
        f_score = 0
        
        if f_List:
            min_food = min([manhattanDistance(newPos, food) for food in f_List])
            f_score = 1.0 / min_food

        for ghState in newGhostStates:
            ghostPos = ghState.getPosition()
            ghostDistance = manhattanDistance(newPos, ghostPos)
            if 0 < ghostDistance < 2:
                gh_score -= 1.0 / ghostDistance

        return successorGameState.getScore() + f_score + gh_score

        
       # return successorGameState.getScore()

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
        _, optimalAction = self.max_val(gameState, 0, 0, self.depth)
        return optimalAction

    def min_val(self, gameState, agentIdx, depthLevel, maxDepthLevel):
        """
        Returns the minimum utility value and the corresponding action
        """
        if depthLevel == maxDepthLevel or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), None

        possibleActions = gameState.getLegalActions(agentIdx)
        minUtilityVal = float('inf')
        chosenAction = None

        for action in possibleActions:
            successorState = gameState.generateSuccessor(agentIdx, action)
            if agentIdx == gameState.getNumAgents() - 1:  # Last ghost's turn
                utility, _ = self.max_val(successorState, 0, depthLevel + 1, maxDepthLevel)
            else:
                utility, _ = self.min_val(successorState, agentIdx + 1, depthLevel, maxDepthLevel)
            if utility < minUtilityVal:
                minUtilityVal = utility
                chosenAction = action

        return minUtilityVal, chosenAction

    def max_val(self, gameState, agentIdx, depthLevel, maxDepthLevel):
        """
        Returns the maximum utility value and the corresponding action
        """
        if depthLevel == maxDepthLevel or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), None

        possibleActions = gameState.getLegalActions(agentIdx)
        maxUtilityVal = float('-inf')
        chosenAction = None

        for action in possibleActions:
            successorState = gameState.generateSuccessor(agentIdx, action)
            utility, _ = self.min_val(successorState, agentIdx + 1, depthLevel, maxDepthLevel)
            if utility > maxUtilityVal:
                maxUtilityVal = utility
                chosenAction = action

        return maxUtilityVal, chosenAction



        
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def maximizeValue(state, agent_id, depth, alpha_bound, beta_bound):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            max_utility = float('-inf')
            available_actions = state.getLegalActions(agent_id)

            for action in available_actions:
                next_state = state.generateSuccessor(agent_id, action)
                max_utility = max(max_utility, minimizeValue(next_state, agent_id + 1, depth, alpha_bound, beta_bound))
                if max_utility > beta_bound:
                    return max_utility
                alpha_bound = max(alpha_bound, max_utility)
            
            return max_utility
        
        def minimizeValue(state, agent_id, depth, alpha_bound, beta_bound):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            min_utility = float('inf')
            next_agent_id = (agent_id + 1) % state.getNumAgents()
            available_actions = state.getLegalActions(agent_id)

            for action in available_actions:
                next_state = state.generateSuccessor(agent_id, action)
                if next_agent_id == 0:
                    min_utility = min(min_utility, maximizeValue(next_state, next_agent_id, depth - 1, alpha_bound, beta_bound))
                else:
                    min_utility = min(min_utility, minimizeValue(next_state, next_agent_id, depth, alpha_bound, beta_bound))
                if min_utility < alpha_bound:
                    return min_utility
                beta_bound = min(beta_bound, min_utility)
            
            return min_utility

        optimal_action = None
        highest_score = float('-inf')
        alpha_bound = float('-inf')
        beta_bound = float('inf')
        available_actions = gameState.getLegalActions()

        for action in available_actions:
            next_state = gameState.generateSuccessor(0, action)
            action_score = minimizeValue(next_state, 1, self.depth, alpha_bound, beta_bound)
            if action_score > highest_score:
                highest_score = action_score
                optimal_action = action
            if highest_score > beta_bound:
                return optimal_action
            alpha_bound = max(alpha_bound, highest_score)
        
        return optimal_action

    
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """ 

    def getAction(self, game_state: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction
        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """

        def pacman_value(state, current_depth):
            if current_depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            max_score = float('-inf')
            actions = state.getLegalActions(0)
            for action in actions:
                next_state = state.generateSuccessor(0, action)
                max_score = max(max_score, ghost_expectation(next_state, 1, current_depth))
            return max_score
        
        def ghost_expectation(state, ghost_index, current_depth):
            if current_depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            expected_value = 0
            actions = state.getLegalActions(ghost_index)
            probability = 1.0 / len(actions)  # Uniform probability for each action
            for action in actions:
                next_state = state.generateSuccessor(ghost_index, action)
                if ghost_index == state.getNumAgents() - 1:
                    expected_value += pacman_value(next_state, current_depth - 1) * probability
                else:
                    expected_value += ghost_expectation(next_state, ghost_index + 1, current_depth) * probability
            return expected_value
        
        available_actions = game_state.getLegalActions()
        optimal_action = available_actions[0]
        highest_score = float('-inf')
        
        for action in available_actions:
            next_state = game_state.generateSuccessor(0, action)
            score = ghost_expectation(next_state, 1, self.depth)
            if score > highest_score:
                highest_score = score
                optimal_action = action
        
        return optimal_action

 

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: This evaluation function considers distances to food, the presence of capsules, and the proximity of ghosts. It calculates a score based on these factors to make optimal decisions for Pacman.
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin():
        return float('inf')  # Pacman wins, highest possible score

    if currentGameState.isLose():
        return float('-inf')  # Pacman loses, lowest possible score

    pacmanLocation = currentGameState.getPacmanPosition()
    food_list = currentGameState.getFood().asList()
    powerPellets = currentGameState.getCapsules()
    ghost_states = currentGameState.getGhostStates()

    # Compute the shortest distance to the nearest food item
    minDistanceToFood = min([euclideanDistance(pacmanLocation, food) for food in food_list], default=float('inf'))

    # Compute the inverse distances to all ghosts
    distancesToGhosts = [euclideanDistance(pacmanLocation, ghost.getPosition()) for ghost in ghost_states]
    
    # Penalize getting close to ghosts
    ghostRisk = sum([10.0 / (distance + 1) if distance < 2 else 0 for distance in distancesToGhosts])

    # Reward eating power pellets
    powerPelletReward = 5 * len(powerPellets)

    # Reward eating more food
    foodReward = 2 * len(food_list)

    # Combine these factors into a final score
    finalScore = currentGameState.getScore() + foodReward + powerPelletReward - ghostRisk - minDistanceToFood

    return finalScore


def euclideanDistance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


# Abbreviation
better = betterEvaluationFunction
