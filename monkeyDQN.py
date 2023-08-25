"""

This is Q-learning with a replay buffer

It is. Garbage.

"""

import gymnasium as gym
import numpy as np

from Model import *
from Dense.Dense import Dense

env = gym.make('CartPole-v1')

state_space = 4
action_space = 2


# def Discrete(state, bins):
#     return tuple(np.digitize(s, b) - 1 for s, b in zip(state, bins))


def Q_Model():
    Q = Model(sqloss, sqdLdA)
    Q.join(Dense(i_size=5, o_size=40, activation="relu", dt = 0.00001))
    Q.join(Dense(o_size=40, activation="relu", dt = 0.00001))
    # Q.join(Dense(o_size=40, activation="relu", dt = 0.00001))
    Q.join(Dense(o_size=1, dt = 0.00001))
    Q.compile()
    return Q

def train(history_x, history_sp, history_r, alpha, gamma, Q):
    size = len(history_x)

    print(size)
    for i in range(10000):
        idx = np.random.randint(size)
        Sa, next, reward = history_x[idx], history_sp[idx], history_r[idx]

        QSa = Q.feed_forward(Sa)

        next[4][0] = 0
        Q0 = Q.feed_forward(next)
        next[4][0] = 1
        Q1 = Q.feed_forward(next)

        maxQ = max(Q0, Q1)
        # if reward != 1:
        #     print("BOIIIIIIIIIIIIIIIIIIIIIII")
        y = (1 - alpha) * QSa + alpha * (reward + gamma * maxQ)

        # if i < 10:
        # print(Sa)
        # print(y)
        Q.cycle(Sa, y)

def Qlearning(episodes=5000000,
              gamma=0.9, alpha=0.1, dt=100, eps=0.2):
    # THIS ALPHA VALUE IS USEFUL. SETTING ALPHA = 1 IS NOT GOOD.
    Q = Q_Model()

    cumulative_reward = 0

    history_x = []  # np.zeros((0, 5, 1))

    history_r = []  # np.zeros((0,))
    history_sp = []  # np.zeros((0, 5, 1))  # s prime

    true_steps = 0

    for step in range(episodes):
        S = env.reset()[0]

        S = np.pad(S, (0, 1), 'constant', constant_values=0).reshape((-1, 1))

        score = 0

        done = False
        while not done:
            if step % dt == 0:
                env.render()
            if np.random.uniform(0, 1) < eps:
                a = env.action_space.sample()
            else:
                # print(S)
                q0 = Q.feed_forward(S)
                S[4][0] = 1
                q1 = Q.feed_forward(S)
                a = int(q1 > q0)

            next, reward, done, *_ = env.step(a)
            reward = not done
            # if done != 1:
            #     print(next) # Bruh... what the fuck is the point of all of this.
            # reward = done
            next = np.pad(next, (0, 1), 'constant', constant_values=0).reshape((-1, 1))
            # print(next)



            S[4][0] = a
            history_x.append(S)

            history_r.append(reward)
            history_sp.append(next)

            true_steps += 1
            # print(true_steps)
            if true_steps % 10000 == 0:


                # Prepare dataset
                history_y = []

                # print("Dataset Preparation")


                # for idx in range(shape):
                #     Sa, next, reward = history_x[idx], history_sp[idx], history_r[idx]
                #
                #     QSa = Q.feed_forward(Sa)
                #
                #     next[4][0] = 0
                #     Q0 = Q.feed_forward(next)
                #     next[4][0] = 1
                #     Q1 = Q.feed_forward(next)
                #
                #
                #     maxQ = max(Q0, Q1)
                #     # if reward != 1:
                #     #     print("BOIIIIIIIIIIIIIIIIIIIIIII")
                #     history_y.append((1 - alpha) * QSa + alpha * (reward + gamma * maxQ))

                # print("Training")
                # Q = Q_Model() # New thing.


                # print("Pre Accuracy")
                # cumulative_loss = 0
                # for Sa, y in zip(history_x, history_y):
                #     _, loss = Q.test(Sa, y)
                #     cumulative_loss += loss

                # print(f"Testing --- Average Loss = {cumulative_loss / shape}")

                train(history_x, history_sp, history_r, alpha, gamma, Q)

                # print("Accuracy")
                # cumulative_loss = 0
                # for Sa, y in zip(history_x, history_y):
                #     _, loss = Q.test(Sa, y)
                #     cumulative_loss += loss

                # print(f"Testing --- Average Loss = {cumulative_loss / shape}")

            S = next

            score += reward
            # print(reward)
        else:
            cumulative_reward += score
            if score > 195 and step >= 100:
                print("-" * 20 + "Solved!" + "-" * 20)

        if step % dt == 0:
            print(cumulative_reward / dt)
            cumulative_reward = 0


if __name__ == "__main__":
    Qlearning()
