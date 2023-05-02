import gym
import os
import numpy as np
import atari_wrappers

class ImageToPyTorch(gym.ObservationWrapper):
    """
    Image shape to channels x weight x height
    """

    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space['image'].shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(old_shape[-1], old_shape[0], old_shape[1]),
            dtype=np.uint8,
        )

    def observation(self, observation):
        print(observation)
        return np.transpose(observation['image'], axes=(2, 0, 1))

if 'MUJOCO_GL' not in os.environ:
    os.environ['MUJOCO_GL'] = 'egl'

def create_env(flags):
    env = gym.make(flags)
    return atari_wrappers.wrap_pytorch(gym.make(flags))

# env = gym.make('memory_maze:MemoryMaze-9x9-ExtraObs-v0')
# env = ImageToPyTorch(env)
# env.reset()

env = create_env('memory_maze:MemoryMaze-9x9-ExtraObs-v0')
env.reset()

env2 = create_env('memory_maze:MemoryMaze-9x9-ExtraObs-v0')

for i in range(5):
    env.step(env.action_space.sample())