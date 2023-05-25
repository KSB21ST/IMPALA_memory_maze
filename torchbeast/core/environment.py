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
from einops import rearrange, repeat


def _format_frame(frame):
    frame = torch.from_numpy(frame)
    return frame.view((1, 1) + frame.shape)  # (...) -> (T,B,...).


def compute_noisy_agent_pos(targets_pos, agent_pos, eps=0.1):
    
    dim_ = agent_pos.dim()
    if dim_ == 3:
        t = targets_pos.size(0)
        targets_pos= rearrange(targets_pos, 't b n c -> (t b) n c')
        agent_pos = repeat(agent_pos, 't b c -> (t b) c')    

    # take targets_pos that are closest to the current agent_pos
    num_tars = targets_pos.size(1)
    assert num_tars >= 3
    if num_tars > 3:
        # agent_pos tb,2
        # targets_pos tb,n,2
        dist = torch.cdist(targets_pos, agent_pos)[..., 0]  # tb,n
        sort_idx = torch.argsort(dist, dim=1)  # tb,n
        targets_pos = targets_pos[torch.arange(targets_pos.size(0)).view(-1, 1), sort_idx][:, :3]  # tb,3,2
        
    agent_pos_rep = repeat(agent_pos, 'tb c -> tb n c', n=3)
    rs = ((targets_pos - agent_pos_rep) ** 2).sum(-1).sqrt() 

    tb = rs.size(0)
    r1 = rs[:, 0] + torch.randn(tb).to(rs.device) * eps    # tb
    r2 = rs[:, 1] + torch.randn(tb).to(rs.device) * eps
    r3 = rs[:, 2] + torch.randn(tb).to(rs.device) * eps
    x1 = targets_pos[:, 0, 0]
    y1 = targets_pos[:, 0, 1]
    x2 = targets_pos[:, 1, 0]
    y2 = targets_pos[:, 1, 1]
    x3 = targets_pos[:, 2, 0]
    y3 = targets_pos[:, 2, 1]
    a = -2*x1 + 2*x2
    b = -2*y1 + 2*y2
    c = r1**2 - r2**2 - x1**2 + x2**2 - y1**2 + y2**2
    d = -2*x2 + 2*x3
    e = -2*y2 + 2*y3
    f = r2**2 - r3**2 - x2**2 + x3**2 - y2**2 + y3**2
    agent_pos_compute = torch.zeros_like(agent_pos).float()
    agent_pos_compute[:, 0] = (c*e-f*b) / (e*a-b*d)
    agent_pos_compute[:, 1] = (c*d-a*f) / (b*d-a*e)

    if dim_ == 3:
        agent_pos_compute = rearrange(agent_pos_compute, '(t b) c -> t b c', t=t)

    return agent_pos_compute


class Environment:
    def __init__(self, gym_env, flags):
        self.gym_env = gym_env
        self.episode_return = None
        self.episode_step = None
        self.flags = flags

    def initial(self):
        # self.flags.device = torch.device(f"cuda:{self.flags.gpu_n}")
        # torch.cuda.set_device(self.flags.device)
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
        self.res = 15
        num_tars = reset_val['targets_pos'].shape[0]
        maze_len = len(reset_val['maze_layout'])
        agent_pos = torch.tensor(reset_val['agent_pos']).float().view(1, 1, 2) 
        agent_ori_pos = agent_pos.clone()
        agent_dir = torch.tensor(reset_val['agent_dir']).float().view(1, 1, 2)
        target_pos = torch.tensor(reset_val['target_pos']).float().view(1, 1, 2) 
        targets_pos = torch.tensor(reset_val['targets_pos']).float().view(1, 1, num_tars, 2)
        if self.flags.posemb == 'noisygt':
            agent_pos = compute_noisy_agent_pos(targets_pos, agent_pos, self.flags.pos_noise)        
        
        # this makes for single grid (1x1) in NxM maze, resolution is 15
        agent_ori_pos = ((agent_ori_pos / maze_len) * (self.res * maze_len)).long().float() / (self.res * maze_len)
        agent_pos = ((agent_pos / maze_len) * (self.res * maze_len)).long().float() / (self.res * maze_len)
        target_pos = ((target_pos / maze_len) * (self.res * maze_len)).long().float() / (self.res * maze_len)
        targets_pos = ((targets_pos / maze_len) * (self.res * maze_len)).long().float() / (self.res * maze_len)

        # print(agent_pos)
        # print(agent_ori_pos)
        # print(target_pos)
        # print(targets_pos)

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
        # self.flags.device = torch.device(f"cuda:{self.flags.gpu_n}")
        # torch.cuda.set_device(self.flags.device)
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
        
        num_tars = reset_val['targets_pos'].shape[0]
        agent_pos = torch.tensor(reset_val['agent_pos']).float().view(1, 1, 2) / maze_len
        agent_dir = torch.tensor(reset_val['agent_dir']).float().view(1, 1, 2)
        target_pos = torch.tensor(reset_val['target_pos']).float().view(1, 1, 2) / maze_len
        targets_pos = torch.tensor(reset_val['targets_pos']).float().view(1, 1, num_tars, 2) / maze_len

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
