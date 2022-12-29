"""

This is Q-learning with a replay buffer

It is. Garbage.

"""

import gymnasium as gym

from Model import *
from Layers.Dense import Dense

env = gym.make('CartPole-v1')

state_space = 4
action_space = 2


# def Discrete(state, bins):
#     return tuple(np.digitize(s, b) - 1 for s, b in zip(state, bins))


def Qlearning(episodes=5000000,
              gamma=0.9, alpha=0.005, dt=100, eps=0.2):
    # THIS ALPHA VALUE IS USEFUL. SETTING ALPHA = 1 IS NOT GOOD.
    Q = Model(sqloss, sqdLdA)
    Q.join(Dense(i_size=5, o_size=20, activation="sigmoid"))
    Q.join(Dense(o_size=20, activation="sigmoid"))
    Q.join(Dense(o_size=20, activation="relu"))
    Q.join(Dense(o_size=1))
    Q.compile()

    cumulative_reward = 0

    window_size = 100
    window_x = np.zeros((window_size, 5, 1))
    window_index = 0

    window_y = np.zeros((window_size, 1, 1))

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
                q0 = Q.feed_forward(np.pad(S, (0, 1), 'constant', constant_values=0).reshape((-1, 1)))
                q1 = Q.feed_forward(np.pad(S, (0, 1), 'constant', constant_values=1).reshape((-1, 1)))
                a = int(q1 > q0)

            next, reward, done, *_ = env.step(a)

            if not done:
                next0 = np.pad(next, (0, 1), 'constant', constant_values=0).reshape((-1, 1))
                next1 = np.pad(next, (0, 1), 'constant', constant_values=1).reshape((-1, 1))
                Sa = np.pad(S, (0, 1), 'constant', constant_values=a).reshape((-1, 1))

                maxQ = np.maximum(Q.feed_forward(next0), Q.feed_forward(next1))

                QSa = Q.feed_forward(Sa)
                # This is what it was supposed to be.
                y = (1 - alpha) * QSa + alpha * (reward + gamma * maxQ)

                # Run it back
                Q.feed_backward(QSa, y)

                window_x[window_index % window_size] = Sa
                window_y[window_index % window_size] = y

                window_index += 1

                for i in range(1):
                    idx = np.random.randint(min(window_index, window_size))
                    Q.cycle(window_x[idx], window_y[idx])

            S = next

            score += reward

        else:
            cumulative_reward += score
            if score > 195 and step >= 100:
                print("Solved!")

        if step % dt == 0:
            print(cumulative_reward / dt)
            cumulative_reward = 0


if __name__ == "__main__":
    Qlearning()
