import tensorflow as tf
import numpy as np
import gym
from collections import deque
from matplotlib import pyplot as plt
import random
from utils import NormalizedActions
ENVS = ['Pendulum-v0',
        'MountainCarContinuous-v0',
        'InvertedPendulum-v1',
        'SemisuperPendulumNoise-v0',
        'SemisuperPendulumRandom-v0']
ENVNUM = 3
if ENVNUM in (0, 3, 4):
    from NAF import NAF
elif ENVNUM == 1:
    from NAF_mc import NAF
elif ENVNUM == 2:
    from NAF_ip import NAF

TRY_NUM = 0
# pure qnaf algorithm 
# directory exp0<env_num><try_num><run_num>

MAX_EP_STEPS = 200
LEARNING_RATE = 0.001
GAMMA = 0.99
TAU = 0.001

RENDER_ENV = False
GYM_MONITOR_EN = True
ENV_NAME = ENVS[ENVNUM]
MONITOR_DIR = './results/exp0' + str(ENVNUM) + str(TRY_NUM)

RANDOM_SEED = 42
BUFFER_SIZE = 800000
MINIBATCH_SIZE = 64

NOISE_MEAN = 0
NOISE_VAR = 1
OU_THETA = 0.15
OU_MU = 0.
OU_SIGMA = 0.3
EXPLORATION_TIME = 50
MAX_EPISODES = 200
if ENVNUM == 2:
    EXPLORATION_TIME = 200
    MAX_EPISODES = 4500


def main(_):
    np.random.seed(RANDOM_SEED)
    tf.set_random_seed(RANDOM_SEED)
    env = NormalizedActions(gym.make(ENV_NAME))
    env.seed(RANDOM_SEED)
    if GYM_MONITOR_EN:
        #it is always better to create n + 1 new dirs than recreate n + 1 old experiments
        if not RENDER_ENV:
            env = gym.wrappers.Monitor(env, MONITOR_DIR, video_callable=False, force=False)
        else:
            env = gym.wrappers.Monitor(env, MONITOR_DIR, force=False)
    with tf.Session() as sess:
        for iteration in range(5):
            monitor_dir = MONITOR_DIR + str(iteration)

            naf = NAF(sess, env, LEARNING_RATE, TAU, GAMMA,
                         BUFFER_SIZE, RANDOM_SEED, monitor_dir,
                         sigma_P_dep = False,
                         det=True, qnaf=True,
                         scope=str(iteration), hn=0, ac=False,
                         sep_V=True)
            naf.run_n_episodes(EXPLORATION_TIME, MAX_EP_STEPS,
                               MINIBATCH_SIZE, explore=True, num_updates=5)
            naf.run_n_episodes(MAX_EPISODES - EXPLORATION_TIME, MAX_EP_STEPS,
                               MINIBATCH_SIZE, explore=False, num_updates=5)
            naf.plot_rewards(monitor_dir)

if __name__ == '__main__':
    tf.app.run()
