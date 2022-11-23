import gymnasium as gym
import numpy as np

from Model import Model
from Dense import Dense


env = gym.make('CartPole-v1')

state_space = 4
action_space = 2

# def Discrete(state, bins):
#     return tuple(np.digitize(s, b) - 1 for s, b in zip(state, bins))


def Qlearning(episodes = 5000,
              gamma = 0.95, alpha = 1, dt = 100, eps = 0.2):

    Q = Model()
    Q.join(Dense(i_size= 5, o_size = 100, activation = "sigmoid"))
    Q.join(Dense(o_size = 100, activation="sigmoid"))
    Q.join(Dense(o_size = 1))
    Q.compile()

    rewards = 0
    for step in range(episodes):
        S = env.reset()[0]

        score = 0

        done = False
        while not done:
            if step % dt == 0:
                env.render()
            if np.random.uniform(0, 1) < eps:
                a = env.action_space.sample()
            else:
                q0 = Q.feed_forward(np.ndarray(S + (0,)).reshape(-1, 1))
                q1 = Q.feed_forward(np.ndarray(S + (1,)).reshape(-1, 1))
                a = q1 > q0

            next, reward, done, *_ = env.step(a)

            if not done:
                Sa = np.ndarray(S + (a,)).reshape(-1, 1)
                QSa = int(Q.feed_forward(Sa))
                next0 = np.ndarray(next + (0,)).reshape(-1, 1)
                next1 = np.ndarray(next + (1,)).reshape(-1, 1)
                maxQ = int(np.maximum(Q.feed_forward(next0), Q.feed_forward(next1)))
                #This is what it was supposed to be.
                y = (1 - alpha) * QSa + alpha * (reward + gamma * maxQ)
            S = next


            score += reward

        else:
            rewards += score
            if score > 195 and step >= 100:
                print("Solved!")

        if step % dt == 0:
            print(rewards / (step + 1))


bin_size = 30

bins = [np.linspace(-4.8, 4.8, bin_size),
           np.linspace(-4, 4, bin_size),
           np.linspace(-0.418, 0.418, bin_size),
           np.linspace(-4, 4, bin_size)]
qtable = np.random.uniform(low = -1, high = 1, size = ([bin_size] * state_space + [action_space]))

Qlearning(qtable, bins)