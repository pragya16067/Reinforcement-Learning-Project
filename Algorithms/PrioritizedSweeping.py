# import all dependencies
import numpy as np
from queue import PriorityQueue

import sys
  
# adding Environments to the system path
sys.path.insert(0, '../.')

from Environments.GridWorld.GridEnv import GridEnv
import matplotlib.pyplot as plt


  

class PrioritizedSweeping():

    def __init__(self, gamma, theta, n, alpha, epsilon):
        # Instantiate a Gridworld environment to perform Prioritized Sweeping
        self.env = GridEnv()                  # Change to gym.make() later
        self.gamma = gamma
        self.theta = theta
        self.n_planning_steps = n
        self.alpha = alpha
        self.epsilon = epsilon
        self.Q_value = np.random.random([self.env.n_states, self.env.n_actions])
        self.PriorityQ = PriorityQueue()
        self.Predecessors = dict()
        self.Model = dict()

    def updatePredecessors(self, s, a, s_prime):
        if(s_prime in self.Predecessors.keys()):
            self.Predecessors[s_prime].append((s,a))
        else:
            self.Predecessors[s_prime] = [(s,a)]

    def choose_best_action(self, state):
        maxQ = np.max(self.Q_value[state])
        maxIdxs = [i for i, val in enumerate(self.Q_value[state]) if(val == maxQ)]
        return np.random.choice(maxIdxs)

    def get_action_from_e_greedy_policy(self, state):
        prob = np.random.rand()
        actions = self.env.actions
        if(prob < self.epsilon):
            return np.random.choice(actions)
        else:
            return self.choose_best_action(state)


    def prioritizedSweepQLearning(self, n_iters):
        n_updates_list = []
        n_episodes_list = []
        mse_list = []

        n_updates = 0
        n_episodes = 0
        for iter in range(n_iters):
            s = self.env.state
            if iter % 250 == 0:
                # print("Iteration: ", iter, "Learning Rate: ", self.alpha)
                if self.alpha > 0.005:
                    self.alpha = 0.75 * self.alpha
                if self.epsilon > 0.1:
                    self.epsilon = self.epsilon - 0.05
                else:
                    self.epsilon = 0.5 * self.epsilon

            if(s == self.env.terminal_state):
                n_episodes += 1
                n_episodes_list.append(n_episodes)
                n_updates_list.append(n_updates)
                n_updates = 0
                print("An episode finished")
                s = self.env.reset()

            a = self.get_action_from_e_greedy_policy(s) # Define Epsilon-Greedy Stochastic Policy
            s_prime, r, done, _ = self.env.step(a)
            self.Model[(s, a)] = [s_prime, r]
            self.updatePredecessors(s, a, s_prime)

            priority = abs(r + self.gamma * np.max(self.Q_value[s_prime]) - self.Q_value[s][a])
            if(priority > self.theta):
                self.PriorityQ.put((-priority, [s, a])) #Insert negative priority since python by default has a minQueue

            for i in range(self.n_planning_steps):
                if(not self.PriorityQ.empty()):
                    p, [s, a] = self.PriorityQ.get()
                    s_prime, r = self.Model[(s, a)]
                    self.Q_value[s][a] = self.Q_value[s][a] + self.alpha*(r + self.gamma * np.max(self.Q_value[s_prime]) - self.Q_value[s][a])
                    n_updates += 1

                    # Loop over all State,Actions that are predicted to reach s
                    if(s in self.Predecessors.keys()):
                        for [pred_s, pred_a] in self.Predecessors[s]:
                            _, pred_reward = self.Model[(pred_s, pred_a)]
                            pred_priority = abs(pred_reward + self.gamma * np.max(self.Q_value[s]) - self.Q_value[pred_s][pred_a])
                            if (pred_priority > self.theta):
                                self.PriorityQ.put((-priority, [pred_s, pred_a]))  # Insert negative priority since this is a minQueue
                else:
                    break

            print("Policy Learned in "+str(iter)+"th iteration:")
            self.prettyPrintPolicy()
            mse_list.append(self.getMSE())

        return n_updates_list, n_episodes_list, mse_list

    def getMSE(self):
        mse = 0
        G = self.env.grid_size
        for i in range(G):
            for j in range(G):
                s = i*G + j
                mse += 1/self.env.n_states * np.absolute(np.sum(self.Q_value[s]) - self.env.optimalValueFunction[i][j])

        return mse

    def prettyPrintPolicy(self):
        m, m = np.shape(self.env.grid)
        for i in range(m):
            for j in range(m):
                if (i == m-1 and j == m-1):
                    print("G", end=" ")
                elif (self.env.grid[i][j]==1): # If Obstacle is detected
                    print("O", end=" ")
                else:
                    s = i*m + j
                    a = np.argmax(self.Q_value[s])
                    if (a == 0):  # AU
                        print(str("\u2191"), end=" ")
                    elif (a == 1):  # AD
                        print(str("\u2193"), end=" ")
                    elif (a == 2):  # AL
                        print(str("\u2190"), end=" ")
                    else:  # AR
                        print(str("\u2192"), end=" ")
            print()

    def plotGraph(self, title, x, y, xLabel, yLabel):
        plt.figure()
        plt.plot(x, y)
        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.show()

if __name__=='__main__':
    np.random.seed(10)
    prSweeping = PrioritizedSweeping(gamma=0.9, theta=0.00001, n=10, alpha=0.2, epsilon=0.3)
    n_iters = 3000
    n_updates_list, n_episodes_list, mse_list = prSweeping.prioritizedSweepQLearning(n_iters=n_iters)
    #print("The Number of updates required for discovering the Optimal Policy are "+str(n_updates_list[-1]))

    prSweeping.plotGraph("", n_episodes_list, n_updates_list, "Episodes", "# of Updates")
    # print(mse_list)
    # print(n_updates_list)
    # print(n_episodes_list)
    prSweeping.plotGraph("MSE vs # of Iterations", np.arange(n_iters), mse_list, "No. of Iterations", "MSE with Optimal Value function")
