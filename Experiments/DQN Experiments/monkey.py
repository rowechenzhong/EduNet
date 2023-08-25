import gymnasium as gym
import numpy as np

env = gym.make('CartPole-v1')

state_space = 4
action_space = 2


def Discrete(state, bins):
    return tuple(np.digitize(s, b) - 1 for s, b in zip(state, bins))


def Qlearning(Q, bins, episodes=50000,
              gamma=0.95, alpha=0.1, dt=100, eps = 0.2):
    rewards = 0
    for step in range(episodes):
        S = Discrete(env.reset()[0], bins)

        score = 0

        done = False
        while not done:
            # if step == -1:
            #     env.render()
            if np.random.uniform(0, 1) < eps:
                a = env.action_space.sample()
            else:
                a = np.argmax(Q[S])
            # print(type(a))
            # print(a)

            k = env.step(a)
            observation, reward, done, *_ = k
            # print(observation)
            # if done == True:
            #     print("-"*100)
            # print(k)
            next = Discrete(observation, bins)

            Q[S + (a,)] = (1 - alpha) * Q[S + (a,)] + alpha * (reward + gamma * np.max(Q[next]))
            S = next

            score += reward


        else:
            rewards += score
            # if score > 195 and step >= 100:
            #     print("Solved!")

        if step % dt == 0:
            print(rewards / dt)
            # print(sorted(map(int, list(Q.reshape((-1,)))))[::100000])
            if rewards / dt > 300:
                break
            rewards = 0

    print("Testing...")
    for step in range(100):
        S = Discrete(env.reset()[0], bins)

        score = 0

        done = False

        while not done:
            a = np.argmax(Q[S])
            k = env.step(a)
            observation, reward, done, *_ = k
            S = Discrete(observation, bins)
            score += reward
            if reward != 1:
                print("Bonk")
            if score % 1000 == 0:
                print(score)
            if score > 100000:
                break
        print("-"* 10 + str(score))


    # print(np.max(Q))
    # print(np.min(Q))
    # Q = Q.reshape((-1,))
    # print(sorted(map(int, list(Q)))[::1000])

bin_size = 30

bins = [np.linspace(-4.8, 4.8, bin_size),
        np.linspace(-4, 4, bin_size),
        np.linspace(-0.418, 0.418, bin_size),
        np.linspace(-4, 4, bin_size)]
qtable = np.random.uniform(low=-1, high=20, size=([bin_size] * state_space + [action_space]))

Qlearning(qtable, bins)


