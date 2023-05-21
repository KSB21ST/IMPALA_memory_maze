# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The environment class for MonoBeast."""

import torch


def _format_frame(frame):
    frame = torch.from_numpy(frame)
    return frame.view((1, 1) + frame.shape)  # (...) -> (T,B,...).


class Environment:
    def __init__(self, gym_env):
        self.gym_env = gym_env
        self.episode_return = None
        self.episode_step = None

    def initial(self):
        initial_reward = torch.zeros(1, 1)
        # This supports only single-tensor actions ATM.
        initial_last_action = torch.zeros(1, 1, dtype=torch.int64)
        self.episode_return = torch.zeros(1, 1)
        self.episode_step = torch.zeros(1, 1, dtype=torch.int32)
        initial_done = torch.ones(1, 1, dtype=torch.uint8)
        
        reset_val = self.gym_env.reset()
        initial_frame = _format_frame(reset_val['image'])
        ''' # additional infos
        image, target_color, agent_pos, agent_dir, targets_vec,
        targets_pos, target_pos, maze_layout
        '''
        maze_len = len(reset_val['maze_layout'])
        agent_pos = torch.tensor(reset_val['agent_pos']).float().view(1, 1, 2) / maze_len
        agent_dir = torch.tensor(reset_val['agent_dir']).float().view(1, 1, 2)
        target_pos = torch.tensor(reset_val['target_pos']).float().view(1, 1, 2) / maze_len
        targets_pos = torch.tensor(reset_val['targets_pos']).float().view(1, 1, 6) / maze_len

        return dict(
            frame=initial_frame,
            reward=initial_reward,
            done=initial_done,
            episode_return=self.episode_return,
            episode_step=self.episode_step,
            last_action=initial_last_action,
            agent_pos=agent_pos,
            agent_dir=agent_dir,
            target_pos=target_pos,
            targets_pos=targets_pos,
        )

    def step(self, action):
        reset_val, reward, done, unused_info = self.gym_env.step(action.item())
        self.episode_step += 1
        self.episode_return += reward
        episode_step = self.episode_step
        episode_return = self.episode_return
        if done:
            frame = self.gym_env.reset()['image']
            self.episode_return = torch.zeros(1, 1)
            self.episode_step = torch.zeros(1, 1, dtype=torch.int32)

        frame = _format_frame(reset_val['image'])
        reward = torch.tensor(reward).view(1, 1)
        done = torch.tensor(done).view(1, 1)
        maze_len = len(reset_val['maze_layout'])
        agent_pos = torch.tensor(reset_val['agent_pos']).float().view(1, 1, 2) / maze_len
        agent_dir = torch.tensor(reset_val['agent_dir']).float().view(1, 1, 2)
        target_pos = torch.tensor(reset_val['target_pos']).float().view(1, 1, 2) / maze_len
        targets_pos = torch.tensor(reset_val['targets_pos']).float().view(1, 1, 6) / maze_len

        return dict(
            frame=frame,
            reward=reward,
            done=done,
            episode_return=episode_return,
            episode_step=episode_step,
            last_action=action,
            agent_pos=agent_pos,
            agent_dir=agent_dir,
            target_pos=target_pos,
            targets_pos=targets_pos,
        )

    def close(self):
        self.gym_env.close()
