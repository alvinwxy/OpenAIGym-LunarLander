import gym
from DQN import Agent
from utils import *
import time
import datetime


def train():
    ######### Hyperparameters #########
    env_name = "LunarLander-v2"
    log_interval = 10       # print avg reward after interval
    gamma = 0.99            # discount for future rewards
    batch_size = 100        # num of transitions sampled from replay buffer
    lr = 0.001
    tau = 0.001
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_dec = 0.11
    max_episodes = 1000     # max num of episodes
    max_timesteps = 2000    # max timesteps in one episode
    directory = "./model"
    filename = "{}_{}".format(env_name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    render_every = 100
    ###################################

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = Agent(lr, state_dim, n_actions)
    replay_buffer = ReplayBuffer(100000)

    # logging variables:
    rewards = []
    best_ep = -200
    best_reward = 0
    log_f = open("log.txt", "w+")
    start_time = time.time()

    # training procedure:
    for episode in range(1, max_episodes+1):

        state = env.reset()
        ep_reward = 0

        for t in range(max_timesteps):

            if episode % render_every == 0:
                env.render()

            # select action
            if np.random.rand() < epsilon:
                action = np.random.randint(0, n_actions)
            else:
                action = agent.choose_action(state)

            # take action in env:
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add((state, action, reward, next_state, float(done)))
            state = next_state

            ep_reward += reward

            agent.learn(replay_buffer, gamma, batch_size, tau)

            if done or t == (max_timesteps-1):
                break

        rewards.append(ep_reward)

        if epsilon > epsilon_min:
            epsilon -= epsilon_dec
        else:
            epsilon = epsilon_min

        avg_reward = np.mean(rewards[-log_interval:])

        print("Episode: {}  \tReward: {:.2f}   \tAverage Reward: {:.3f}   \tTime Steps: {}"
              .format(episode, ep_reward, avg_reward, t))

        # logging updates:
        log_f.write('{},{}\n'.format(episode, ep_reward))
        log_f.flush()

        if ep_reward > best_reward:
            best_reward = ep_reward
            best_ep = episode

        if ep_reward > 200:
            if test(env, agent):
                print("################################### Solved!! ###################################")
                break

    n_episode = [i for i in range(episode)]
    fig = filename + ".png"
    plot(n_episode, rewards, filename=fig)
    end_time = time.time()
    training_time = end_time - start_time
    training_sec = training_time % 60
    training_min = (training_time / 60) % 60
    training_hr = (training_time / 60) / 60
    print("Training Time: {:.0f} hr {:.0f} min {:.0f} sec".format(training_hr, training_min, training_sec))
    print("Best episode: {}   \tBest episode reward: {:.3f}".format(best_ep, best_reward))


def test(env, agent):

    test_episodes = 100
    solved = False
    total_reward = 0

    print("Testing model", end='')

    for episodes in range(test_episodes):

        state = env.reset()
        done = False

        while not done:

            action = agent.choose_action(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward

        if episodes % 10 == 0:
            print(".", end='')

    avg_reward = total_reward / test_episodes
    print('Testing average reward : ', avg_reward)

    if avg_reward >= 200:
        solved = True

    return solved


if __name__ == '__main__':
    train()
