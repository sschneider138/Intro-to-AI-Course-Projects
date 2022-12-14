# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import collections

import mdp
import util
from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
    * Please read learningAgents.py before reading this.*

    A ValueIterationAgent takes a Markov decision process
    (see mdp.py) on initialization and runs value iteration
    for a given number of iterations using the supplied
    discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
        Your value iteration agent should take an mdp on
        construction, run the indicated number of iterations
        and then act according to the resulting policy.

        Some useful mdp methods you will use:
            mdp.getStates()
            mdp.getPossibleActions(state)
            mdp.getTransitionStatesAndProbs(state, action)
            mdp.getReward(state, action, nextState)
            mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        for iteration in range(0, self.iterations):
            qValues = self.values.copy()
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    continue
                optimalMove = self.computeActionFromValues(state)
                qValue = self.computeQValueFromValues(state, optimalMove)
                qValues[state] = qValue
            self.values = qValues

    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
        Compute the Q-value of action in state from the
        value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        transition = self.mdp.getTransitionStatesAndProbs(state, action)
        qValue = 0
        for futureState, probs in transition:
            futureReward = self.mdp.getReward(state, action, futureState)
            discount = self.discount
            futureQVal = self.getValue(futureState)
            qValue += probs * (futureReward + discount * futureQVal)
        return qValue

    def computeActionFromValues(self, state):
        """
        The policy is the best action in the given state
        according to the values currently stored in self.values.

        You may break ties any way you see fit.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None
        else:
            possibleActions = self.mdp.getPossibleActions(state)
            actionQ = util.Counter()
            for possibleAction in possibleActions:
                actionQ[possibleAction] = self.computeQValueFromValues(state, possibleAction)
            return actionQ.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
    * Please read learningAgents.py before reading this.*

    An AsynchronousValueIterationAgent takes a Markov decision process
    (see mdp.py) on initialization and runs cyclic value iteration
    for a given number of iterations using the supplied
    discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=1000):
        """
        Your cyclic value iteration agent should take an mdp on
        construction, run the indicated number of iterations,
        and then act according to the resulting policy. Each iteration
        updates the value of only one state, which cycles through
        the states list. If the chosen state is terminal, nothing
        happens in that iteration.

        Some useful mdp methods you will use:
            mdp.getStates()
            mdp.getPossibleActions(state)
            mdp.getTransitionStatesAndProbs(state, action)
            mdp.getReward(state)
            mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        mdpStates = self.mdp.getStates()
        i = 0
        for iteration in range(0, self.iterations):
            if i == len(mdpStates): i = 0
            targetState = mdpStates[i]
            i += 1
            if self.mdp.isTerminal(targetState):
                continue
            else:
                bestAction = self.computeActionFromValues(targetState)
                qValue = self.computeQValueFromValues(targetState, bestAction)
                self.values[targetState] = qValue


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
    * Please read learningAgents.py before reading this.*

    A PrioritizedSweepingValueIterationAgent takes a Markov decision process
    (see mdp.py) on initialization and runs prioritized sweeping value iteration
    for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
        Your prioritized sweeping value iteration agent should take an mdp on
        construction, run the indicated number of iterations,
        and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def maxQValue(self, state):
        '''function to compute maximum Q-value across all possible action from a state'''
        qv = []
        axn = self.mdp.getPossibleActions(state)
        if(len(axn) == 0):
            return self.values[state]
        for action in axn:
            qv.append(self.computeQValueFromValues(state, action))
        return max(qv)
        
    def runValueIteration(self):
        # getting states using getStates()
        states = self.mdp.getStates()
        pred = {}
        for s in states:
            pred[s] = set()
        for s in states:
            actions = self.mdp.getPossibleActions(s)
            for x in actions:
                trans = self.mdp.getTransitionStatesAndProbs(s, x)
                for next_state, _ in trans:
                    pred[next_state].add(s)

        pq = util.PriorityQueue()

        s = self.mdp.getStates()
        for x in s:
            if(x == "TERMINAL_STATE"):
                continue
            max_q = self.maxQValue(x)
            # computing the difference
            diff = abs(self.values[x] - max_q)
            pq.push(x, -diff)
        iter = 0
        while (iter < self.iterations):
            if(pq.isEmpty()):
                return
            state = pq.pop()
            self.values[state] = self.maxQValue(state)
            for p in pred[state]:
                max_q = self.maxQValue(p)
                diff = abs(self.values[p] - max_q)
                if(diff > self.theta):
                    pq.update(p, -diff)
            iter += 1

        return
