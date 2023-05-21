import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat


def compute_noisy_agent_pos(targets_pos, agent_pos, eps=0.1):
    
    dim_ = agent_pos.dim()
    if dim_ == 3:
        t = targets_pos.size(0)
        targets_pos= rearrange(targets_pos, 't b c -> (t b) c')
        agent_pos = repeat(agent_pos, 't b c -> (t b) c')    

    targets_pos= rearrange(targets_pos, 'tb (n c) -> tb n c', n=3)
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


class resnet(nn.Module):
    def __init__(self, in_ch, num_ch, num_blocks):
        super(resnet, self).__init__()
        self._conv = nn.Conv2d(in_channels=in_ch,
                              out_channels=num_ch,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self._max_pool = nn.MaxPool2d(kernel_size=3,
                                      stride=2,
                                      padding=1)
        
        self._res_convs0 = nn.ModuleList([nn.Conv2d(in_channels=num_ch,
                                                    out_channels=num_ch,
                                                    kernel_size=3,
                                                    stride=1,
                                                    padding=1) for i in range(num_blocks)])
        self._res_convs1 = nn.ModuleList([nn.Conv2d(in_channels=num_ch,
                                                    out_channels=num_ch,
                                                    kernel_size=3,
                                                    stride=1,
                                                    padding=1) for i in range(num_blocks)])

    def forward(self, conv_out):
        # Downscale.
        conv_out = self._conv(conv_out)
        conv_out = self._max_pool(conv_out)
        # Residual block(s).
        for (res_conv0, res_conv1) in zip(self._res_convs0, self._res_convs1):
            block_input = conv_out
            conv_out = F.relu(conv_out)
            conv_out = res_conv0(conv_out)
            conv_out = F.relu(conv_out)
            conv_out = res_conv1(conv_out)
            conv_out += block_input

        return conv_out


class AtariNet(nn.Module):
    def __init__(self, observation_shape, num_actions, flags):
        super(AtariNet, self).__init__()
        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.posemb = flags.posemb
        self.pos_noise = flags.pos_noise

        # Feature extraction.
        enc_modules = nn.ModuleList([resnet(in_ch, num_ch, num_blocks) 
                                     for in_ch, num_ch, num_blocks in [(3, 16, 2), (16, 32, 2), (32, 32, 2)]])
        self.enc = nn.Sequential(*enc_modules)

        # Fully connected layer.
        self.fc = nn.Linear(2048, 256)

        # FC output size + one-hot of last action + last reward.
        if self.posemb == 'noemb':
            core_output_size = self.fc.out_features + num_actions + 1
        # FC output size + one-hot of last action + last reward + agent_pos + agent_dir + target_pos
        elif self.posemb == 'gt':
            core_output_size = self.fc.out_features + num_actions + 1 + 6
        elif self.posemb == 'noisygt':
            core_output_size = self.fc.out_features + num_actions + 1 + 6

        self.use_lstm = flags.use_lstm
        if self.use_lstm:
            self.core = nn.LSTM(core_output_size, core_output_size, 2)

        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = nn.Linear(core_output_size, 1)

    def initial_state(self, batch_size):
        if not self.use_lstm:
            return tuple()
        return tuple(
            torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size)
            for _ in range(2)
        )

    def forward(self, inputs, core_state=()):

        x = inputs["frame"]  # [T, B, C, H, W].
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = x.float() / 255.0
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        x = self.enc(x)
        x = x.view(T * B, -1)
        x = F.relu(self.fc(x))

        one_hot_last_action = F.one_hot(
            inputs["last_action"].view(T * B), self.num_actions
        ).float()
        clipped_reward = torch.clamp(inputs["reward"], -1, 1).view(T * B, 1)
        agent_pos = inputs['agent_pos'].view(T * B, 2)
        agent_dir = inputs['agent_dir'].view(T * B, 2)
        target_pos = inputs['target_pos'].view(T * B, 2)
        targets_pos = inputs['targets_pos'].view(T * B, 6)
        
        # concat pos info depending on posemb option
        if self.posemb == 'noemb':
            core_input = torch.cat([x, clipped_reward, one_hot_last_action], dim=-1)
        elif self.posemb == 'gt':
            core_input = torch.cat([x, clipped_reward, one_hot_last_action,
                                     agent_pos, agent_dir, target_pos], dim=-1)
        elif self.posemb == 'noisygt':
            # calculate noisy agent_pos
            noisy_agent_pos = compute_noisy_agent_pos(targets_pos, agent_pos, self.pos_noise)
            core_input = torch.cat([x, clipped_reward, one_hot_last_action,
                                     noisy_agent_pos, agent_dir, target_pos], dim=-1)
                

        if self.use_lstm:
            core_input = core_input.view(T, B, -1)
            core_output_list = []
            notdone = (~inputs["done"]).float()
            for input, nd in zip(core_input.unbind(), notdone.unbind()):
                # Reset core state to zero whenever an episode ended.
                # Make `done` broadcastable with (num_layers, B, hidden_size)
                # states:
                nd = nd.view(1, -1, 1)
                core_state = tuple(nd * s for s in core_state)
                output, core_state = self.core(input.unsqueeze(0), core_state)
                core_output_list.append(output)
            core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        else:
            core_output = core_input
            core_state = tuple()

        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return (
            dict(policy_logits=policy_logits, baseline=baseline, action=action),
            core_state,
        )

