# myTeam.py
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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'DummyAgent', second = 'DummyAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''
    self.start = gameState.getAgentPosition(self.index)

    # Calculate the boundary x coordinate of two teams
    self.boundaryOfTeam = (gameState.data.layout.width - 2) // 2
    if self.red is False:
      self.boundaryOfTeam += 1    


  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''
    values = [self.evaluate(gameState, a) for a in actions] # Calculate value for legal Actions
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue] # choose best Actions

    return random.choice(bestActions)
  

  # Generate next state for particular action
  def getSuccessor(self, gameState, action):
    successor = gameState.generateSuccessor(self.index, action)
    return successor  
  

  # Return value of linear combination of features and weights
  def evaluate(self, gameState, action):
    # Get the score diff
    winningScore = self.getScore(gameState)
    myState = gameState.getAgentState(self.index)
    myPos = myState.getPosition()

    enemies = [gameState.getAgentState(a) for a in self.getOpponents(gameState)]
    enemydists = [self.getMazeDistance(myPos, enemy.getPosition()) for enemy in enemies]
    enemydist = min(enemydists)

    # if score diff is larger than 3, or Agent carries more than 3 foods, activate Defensive Mode
    if winningScore > 0 or gameState.getAgentState(self.index).numCarrying > 1 or (enemydist < 3 and not myState.isPacman):
      features = self.defensiveModeFeatures(gameState, action)
      weights = self.defensiveModeWeights()
    # else, get foods at enemy's area
    else:
      features = self.offensiveModeFeatures(gameState, action)
      weights = self.offensiveModeWeights()

    return features * weights

  
  def defensiveModeFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # calculate the closest location and distance to the boundary
    features['closestBoundaryDist'] = 9999

    for i in range(successor.data.layout.height):
      if not successor.hasWall(self.boundaryOfTeam, i):
        dist = self.getMazeDistance(myPos, (self.boundaryOfTeam, i))
        if dist < features['closestBoundaryDist']:
          features['closestBoundaryDist'] = dist

    # calculate the closest ghost's distance
    opponents = [successor.getAgentState(a) for a in self.getOpponents(successor)]
    activeGhosts = [ghost for ghost in opponents if not ghost.isPacman and ghost.getPosition() is not None and ghost.scaredTimer == 0]
    
    features['closestGhostDist'] = 0

    if len(activeGhosts) > 0:
      ghostDists = [self.getMazeDistance(myPos, ghost.getPosition()) for ghost in activeGhosts]
      features['closestGhostDist'] = 1 / (min(ghostDists) * min(ghostDists))  # if dist is lower, more dangerous

    # if Agent is in its area, ghostdist and boundarydist is useless
    if not myState.isPacman:
      features['closestGhostDist'] = 0
      features['closestBoundaryDist'] = 0
      features['isGhost'] = 1

    # calculate x coordinate difference to the boundary
    features['boundXCordinate'] = abs(myPos[0] - self.boundaryOfTeam)

    # calculate the closest pacman's distance
    activePacmans = [pacman for pacman in opponents if pacman.isPacman and pacman.getPosition() is not None]

    features['closestPacmanDist'] = 0
    features['pacmanNum'] = len(activePacmans)
    if len(activePacmans) > 0:
      pacmanDists = [self.getMazeDistance(myPos, pacman.getPosition()) for pacman in activePacmans]
      features['closestPacmanDist'] = min(pacmanDists)  # if dist is lower, more dangerous

    if myState.isPacman:
      features['closestPacmanDist'] = 0
      features['boundaryXCordinate'] = 0
      features['isGhost'] = 0
      features['pacmanNum'] = 0

    return features
  

  def defensiveModeWeights(self):
    return {'closestBoundaryDist': -1, "closestGhostDist": -10, "isGhost": 10000000, "closestPacmanDist": -5, "boundaryXCordinate": -1, "pacmanNum": -100}
  

  def offensiveModeFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['pacman'] = 0

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    foods = self.getFood(successor).asList()
    features['foodNum'] = len(foods)
    features['distToFood'] = 9999

    for food in foods:
      dist = self.getMazeDistance(myPos, food)
      if dist < features['distToFood']:
        features['distToFood'] = dist

    opponents = [successor.getAgentState(a) for a in self.getOpponents(successor)]
    activeGhosts = [ghost for ghost in opponents if not ghost.isPacman and ghost.getPosition() is not None and ghost.scaredTimer == 0]
    
    features['closestGhostDist'] = 0

    if len(activeGhosts) > 0:
      ghostDists = [self.getMazeDistance(myPos, ghost.getPosition()) for ghost in activeGhosts]
      features['closestGhostDist'] = 1 / (min(ghostDists) * min(ghostDists))  # if dist is lower, more dangerous

    if myState.isPacman:
      features['pacman'] = 1

    teams = [successor.getAgentState(a) for a in self.getTeam(successor)]
    teamPos = [team.getPosition() for team in teams]
    teamDist = self.getMazeDistance(teamPos[0], teamPos[1])
    if teamDist == 0:
      features['teamDist'] = 1
    else:
        features['teamDist'] = 1 / teamDist

    return features
  

  def offensiveModeWeights(self):
    return {"closestGhostDist": -50, 'distToFood': -1, "foodNum": -100, "pacman": 1000, "teamDist": -16}

    

