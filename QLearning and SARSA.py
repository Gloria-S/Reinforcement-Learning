import gym
import numpy as np
from matplotlib import pyplot as plt


class Arguments:
    def __init__(self):
        self.env = None
        self.obs_n = None
        self.act_n = None
        self.agent = None

        # Set your parameters here
        self.episodes =800
        self.max_step =100
        self.lr =0.1
        self.gamma =0.95
        self.epsilon =0.1


class QLearningAgent:
    def __init__(self, args):
        self.obs_n = args.obs_n
        self.act_n = args.act_n
        self.lr = args.lr
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.Q = np.zeros((args.obs_n, args.act_n))

    def select_action(self, obs, if_train=True):
        # Implement your code here
        if if_train:
            if np.random.uniform(0, 1) < self.epsilon:  # 随机选择动作
                action = np.random.choice(self.act_n)
            else:  # 根据table的Q值选择动作
                action_list = np.where(self.Q[obs, :] == np.max(self.Q[obs, :]))[0]
                # print(action_list)
                action = np.random.choice(action_list)
        else:
            action=np.argmax(self.Q[obs])
        return action

    def update(self, transition):
        obs, action, reward, next_obs, done = transition
        # Implement your code here
        predict_Q=self.Q[obs,action]
        if done:
            target_Q=reward
        else:
            target_Q=reward+self.gamma*np.max(self.Q[next_obs,:])
        self.Q[obs,action]+=self.lr*(target_Q-predict_Q)


class SARSAAgent:
    def __init__(self, args):
        self.obs_n = args.obs_n
        self.act_n = args.act_n
        self.lr = args.lr
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.Q = np.zeros((args.obs_n, args.act_n))

    def select_action(self, obs, if_train=True):
        # Implement your code here
        if if_train:
            if np.random.uniform(0, 1) < self.epsilon:  # 随机选择动作
                action = np.random.choice(self.act_n)
            else:  # 根据table的Q值选择动作
                action_list = np.where(self.Q[obs, :] == np.max(self.Q[obs, :]))[0]
                # print(action_list)
                action = np.random.choice(action_list)
        else:
            action=np.argmax(self.Q[obs])

        return action

    def update(self, transition):
        obs, action, reward, next_obs, next_action, done = transition
        # Implement your code here
        predict_Q=self.Q[obs,action]
        if done:
            target_Q=reward
        else:
            target_Q=reward+self.gamma*self.Q[next_obs,next_action]
        self.Q[obs,action]+=self.lr*(target_Q-predict_Q)


def q_learning_train(args):
    env = args.env
    agent = args.agent
    episodes = args.episodes
    max_step = args.max_step
    rewards = []
    mean_100ep_reward = []
    for episode in range(episodes):
        episode_reward = 0
        # Implement your code here
        obs=env.reset()
        for t in range(max_step):
            # Implement your code here
            action=agent.select_action(obs)
            next_obs, reward, done, _ = env.step(action)
            transition=obs, action, reward, next_obs, done
            agent.update(transition)
            obs=next_obs
            episode_reward += reward
            if done: break
        print(f'Episode {episode}\t Step {t}\t Reward {episode_reward}')
        rewards.append(episode_reward)
        if len(rewards) < 100:
            mean_100ep_reward.append(np.mean(rewards))
        else:
            mean_100ep_reward.append(np.mean(rewards[-100:]))
    return mean_100ep_reward


def sarsa_train(args):
    env = args.env
    agent = args.agent
    episodes = args.episodes
    max_step = args.max_step
    rewards = []
    mean_100ep_reward = []
    for episode in range(episodes):
        episode_reward = 0
        # Implement your code here
        obs=env.reset()
        action=agent.select_action(obs)
        for t in range(max_step):
            # Implement your code here
            next_obs,reward,done,_=env.step(action)
            next_action=agent.select_action(next_obs)
            transition=obs, action, reward, next_obs, next_action, done
            agent.update(transition)
            action=next_action
            obs=next_obs
            episode_reward += reward
            if done: break
        print(f'Episode {episode}\t Step {t}\t Reward {episode_reward}')
        rewards.append(episode_reward)
        if len(rewards) < 100:
            mean_100ep_reward.append(np.mean(rewards))
        else:
            mean_100ep_reward.append(np.mean(rewards[-100:]))
    return mean_100ep_reward


def q_learning_test(args):
    # Implement your code here
    env = args.env
    agent = args.agent
    total_reward=0
    obs=env.reset()
    action_list=[]
    print("Q_learning test:")
    while True:
        action=agent.select_action(obs,if_train=False)
        next_obs, reward, done, _ = env.step(action)
        # print(obs,action,next_obs)
        total_reward+=reward
        obs=next_obs
        action_list.append(action)
        # env.render()
        if done:
            break
    print("total reward:",total_reward)
    print(action_list)

    action_table = [["" for i in range(12)] for i in range(4)]
    for obs in range(args.obs_n):
        action = agent.select_action(obs, False)
        i = int(obs / 12)
        j = obs - 12 * int(obs / 12)
        action_table[i][j] = action
    print("q_learning action")
    for line in range(4):
        print(action_table[line])


def sarsa_test(args):
    # Implement your code here
    env=args.env
    agent=args.agent
    total_reward=0
    action_list=[]
    obs=env.reset()
    print("SARSA learning test:")
    while True:
        action = agent.select_action(obs,False)
        # print(obs,action)
        next_obs,reward,done,_=env.step(action)
        # print(next_obs)
        total_reward+=reward
        obs=next_obs
        action_list.append(action)
        # env.render()
        if done:
            break
    print("total reward:",total_reward)
    print(action_list)

    action_table=[["" for i in range(12)]for i in range(4)]
    for obs in range(args.obs_n):
        action=agent.select_action(obs,False)
        i=int(obs/12)
        j=obs-12*int(obs/12)
        action_table[i][j]=action
    print("sarsa action")
    for line in range(4):
        print(action_table[line])

if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)

    q_learning_args = Arguments()
    env = gym.make("CliffWalking-v0")
    q_learning_args.env = env
    q_learning_args.obs_n = env.observation_space.n
    q_learning_args.act_n = env.action_space.n
    q_learning_args.agent = QLearningAgent(q_learning_args)

    sarsa_args = Arguments()
    env = gym.make("CliffWalking-v0")
    sarsa_args.env = env
    sarsa_args.obs_n = env.observation_space.n
    sarsa_args.act_n = env.action_space.n
    sarsa_args.agent = SARSAAgent(sarsa_args)

    q_learning_rewards = q_learning_train(q_learning_args)
    sarsa_rewards = sarsa_train(sarsa_args)

    q_learning_test(q_learning_args)
    sarsa_test(sarsa_args)

    plt.plot(range(q_learning_args.episodes), q_learning_rewards, label='Q Learning')
    plt.plot(range(sarsa_args.episodes), sarsa_rewards, label='SARSA')
    plt.legend()
    plt.show()
