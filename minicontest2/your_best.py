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
               first='CooperativeAgent', second='CooperativeAgent'):
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

class CooperativeAgent(CaptureAgent):
    # 각 에이전트가 목표로 가고 있는 food를 표시해 같은 food를 향하지 않도록 하기 위한 변수
    currentGoals = dict()

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

        # 자신의 팀 영역의 경계가 되는 X 좌표 계산
        self.boundaryX = (gameState.data.layout.width - 2) // 2

        if not self.red:
            self.boundaryX += 1

        # 자신의 팀 영역의 경계가 되는 모든 좌표 계산
        self.boundaries = []
        for i in range(gameState.data.layout.height):
            if not gameState.hasWall(self.boundaryX, i):
                self.boundaries.append((self.boundaryX, i))

    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        actions = gameState.getLegalActions(self.index)

        values = [self.evaluate(gameState, action) for action in actions]

        maxValue = max(values)
        bestActions = [action for action, value in zip(actions, values) if value == maxValue]

        return random.choice(bestActions)

    def getSuccessor(self, gameState, action):
        # 현재 상태에서 특정한 액션을 취했을 때 다음 상태를 생성
        successor = gameState.generateSuccessor(self.index, action)
        return successor

    def evaluate(self, gameState, action):
        # 에이전트에 따라 방어하는 food의 개수가 15개 또는 10개 이하가 되면 수비 모드로 전환
        leftFood = 10 if self.index == self.getTeam(gameState)[0] else 15

        # 현재 우리 팀의 food가 일정 개수 이하가 된다면 팀 영역으로 돌아와 수비 모드로 전환
        if len(self.getFoodYouAreDefending(gameState).asList()) < leftFood:
            features = self.getDefensiveFeatures(gameState, action)
            weights = self.getDefensiveWeights()

        # 현재 우리 팀의 food가 15개 넘게 남아 있다면 상대편 진영에서 food를 모음
        else:
            # 어떤 액션의 가치를 feature * weights의 linear combination으로 계산
            features = self.getOffensiveFeatures(gameState, action)
            weights = self.getOffensiveWeights()

        return features * weights

    def getOffensiveFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # 현재 먹은 food가 2개 이상이라면 팀 영역으로 돌아가 점수 올리도록 feature 설정
        if gameState.getAgentState(self.index).numCarrying >= 3:
            features['distToHome'] = min([self.getMazeDistance(myPos, boundary) for boundary in self.boundaries])

        else:
            foods = self.getFood(successor).asList()
            features['foodLeft'] = len(foods)
            features['distToFood'] = 9999

            for food in foods:
                # 가장 가까운 food가 다른 에이전트와 겹치면 다음으로 가까운 food를 선택
                if (self.index + 2) % 4 in self.currentGoals and food == self.currentGoals[(self.index + 2) % 4]:
                    continue

                dist = self.getMazeDistance(myPos, food)
                if dist < features['distToFood']:
                    features['distToFood'] = dist
                    self.currentGoals[self.index] = food

        # 가장 가까운 ghost까지의 거리 계산
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        ghosts = [enemy for enemy in enemies if
                  not enemy.isPacman and enemy.getPosition() is not None and enemy.scaredTimer == 0]

        if len(ghosts) > 0:
            ghostsDist = [self.getMazeDistance(myPos, ghost.getPosition()) for ghost in ghosts]
            # 가장 가까운 ghost까지의 거리가 멀 때는 큰 영향을 끼치지 않지만 가까워질수록 그 영향이 커지도록 반비례 관계
            features['distToGhost'] = 1 / min(ghostsDist)

            # 어떤 행동의 결과 고스트에게 잡혀 원래 위치로 돌아가게 된다면 그 방향으로 가지 않게 큰 패널티를 줌
            if myPos == self.start:
                features['distanceToGhost'] = 9999

            # food가 있는 곳이 3면이 막힌 터널이고, 고스트가 3번의 이동 안에 도착할 수 있으면
            # 그 터널 속 food는 먹지 않도록 feature 설정
            wallCount = 0
            for i in [-1, 1]:
                if successor.hasWall(int(myPos[0] + i), int(myPos[1])):
                    wallCount += 1
                if successor.hasWall(int(myPos[0]), int(myPos[1] + i)):
                    wallCount += 1

            if wallCount == 3 and min(ghostsDist) <= 3:
                features['isTunnel'] = 1

        # 아직 3개 이상 food를 carry하지 않고 있더라도 고스트를 피해 도망다니던 중 영역 경계에 도달하면
        # 팀 영역으로 들어가 점수를 올리도록 feature 설정
        features['successorScore'] = self.getScore(successor)

        return features

    def getOffensiveWeights(self):
        # 터널인 경우 foodLeft가 -1 돼서 전체적으로 +100 되는 것을 상쇄시키고
        # 그것보다 더 패널티를 주어야 하므로 -100 보다 더 작게 weight 설정
        return {'foodLeft': -100, 'distToHome': -1, 'distToFood': -1, 'distToGhost': -15, 'isTunnel': -110, 'successorScore': 10}

    def getDefensiveFeatures(self, gameState, action):
        # 공격 모드일 때 목표했던 가장 가까운 food 목표 해제
        if self.index in self.currentGoals:
            del self.currentGoals[self.index]

        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # 액션을 하고 난 후의 invader의 수
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [enemy for enemy in enemies if enemy.isPacman and enemy.getPosition() != None]
        features["numInvaders"] = len(invaders)

        # invader가 존재한다면 가장 가까운 invader 까지의 거리
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        # 액션을 취하고 난 후 invader가 없고, 경계로 부터 멀리 떨어져 있을 경우 가장 가까운 경계까지의 거리
        else:
            deltaXCoord = self.boundaryX - myPos[0] if self.red else myPos[0] - self.boundaryX
            if deltaXCoord >= 4 or deltaXCoord < 0:
                features['nearBoundary'] = min([self.getMazeDistance(myPos, boundary) for boundary in self.boundaries])

        # food를 모으던 중 수비 모드로 전환되어서 팀 영역으로 돌아갈 때 고스트가 일정 거리 안으로 들어오는 방향으로
        # 가지 않도록하는 feature
        if gameState.getAgentState(self.index).isPacman:
            ghosts = [enemy for enemy in enemies if
                      not enemy.isPacman and enemy.getPosition() is not None and enemy.scaredTimer == 0]
            if len(ghosts) > 0:
                dists = [self.getMazeDistance(myPos, ghost.getPosition()) for ghost in ghosts]
                if min(dists) <= 2 or myPos == self.start:
                    features['nearGhost'] = 1

        return features

    def getDefensiveWeights(self):
        # invader의 수/ invader까지의 거리 / boundary까지의 거리는 작을수록 좋으므로 음의 weight
        # 고스트의 주변으로 가지 않도록 nearGhost는 -infinity
        return {'numInvaders': -100, 'invaderDistance': -1, 'nearBoundary': -1, 'nearGhost': -9999}
