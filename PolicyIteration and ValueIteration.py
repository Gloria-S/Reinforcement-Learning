import gym
import numpy as np
from matplotlib import pyplot as plt


class CliffWalking:
    def __init__(self):
        self.actions = (0, 1, 2, 3)
        self.rewards = [[-1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1, -1],
                        [-1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1, -1],
                        [-1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1, -1],
                        [-1, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,  0]]

    def step(self, pos, a):
        i, j = pos
        if a == 0: # up
            i_ = i-1 if i > 0 else 0
            j_ = j
        elif a == 1: # right
            i_ = i
            j_ = j+1 if j < 11 else j
        elif a == 2: # down
            i_ = i+1 if i < 3 else i
            j_ = j
        elif a == 3: # left
            i_ = i
            j_ = j-1 if j > 0 else j
        return i_, j_, self.rewards[i_][j_]


class PolicyIteration:
    def __init__(self, env):
        self.env = env
        self.PI = np.array([[np.random.choice((0,1,2,3)) for j in range(12)] for j in range(4)])
        self.V = np.array([[np.random.random() for j in range(12)] for j in range(4)])
        self.V[-1][-1] = 0
        self.gamma=0.9

    def policy_evaluate(self):
        err=0.00001
        for _ in range(1000):
            max_err=0.0
            for i in range(4):
                for j in range(12):
                    if i==3 and j==11:
                        continue
                    next_row,next_col,reward=self.env.step((i,j),self.PI[i][j])
                    old_v=self.V[i][j]
                    self.V[i][j]=reward+self.gamma*self.V[next_row][next_col]
                    abs_err=abs(self.V[i][j]-old_v)
                    max_err=abs_err if abs_err >max_err else max_err
            if max_err<err:
                break

    def policy_improve(self):
        changed=False
        for i in range(4):
            for j in range(12):
                if i==3 and j==11:
                    continue
                max_v_a=self.PI[i][j]
                max_v=-10e6
                for action in range(3):
                    next_row,next_col,reward=self.env.step((i,j),action)
                    q_reward=reward+self.gamma*self.V[next_row][next_col]
                    if q_reward>max_v:
                        max_v_a=action
                        max_v=q_reward
                if self.PI[i][j]!=max_v_a:
                    changed=True
                self.PI[i][j]=max_v_a
        return changed

    def learn(self):
        # Implement your code here
        not_changed_count=0
        for i in range(10000):
            self.policy_evaluate()
            changed=self.policy_improve()
            if changed:
                not_changed_count=0
            else:
                not_changed_count+=1
            if not_changed_count==10:
                break
        print(self.PI)


class ValueIteration:
    def __init__(self, env):
        self.env = env
        self.PI = np.array([[np.random.choice((0,1,2,3)) for j in range(12)] for j in range(4)])
        self.V = np.array([[np.random.random() for j in range(12)] for j in range(4)])
        self.V[-1][-1] = 0
        self.gamma=0.9

    def value_iteration(self):
        err=0.0
        for i in range(4):
            for j in range(12):
                if i==3 and j==11:
                    continue
                action=self.PI[i][j]
                next_row, next_col, reward = self.env.step((i, j), self.PI[i][j])
                new_value=reward+self.gamma*self.V[next_row][next_col]
                new_action=action
                for a in range(4):
                    next_row, next_col, reward = self.env.step((i, j), a)
                    if new_value<reward+self.V[next_row][next_col]:
                        new_value=reward+self.gamma*self.V[next_row][next_col]
                        new_action=a
                err=max(err,abs(new_value-self.V[i][j]))
                self.V[i][j]=new_value
                self.PI[i][j]=new_action
        return err

    def learn(self):
        # Implement your code here
        for i in range(10000):
            error=self.value_iteration()
            if error<0.00001:
                break
        print(self.PI)


if __name__ == '__main__':
    np.random.seed(0)
    env = CliffWalking()

    PI = PolicyIteration(env)
    PI.learn()

    VI = ValueIteration(env)
    VI.learn()
