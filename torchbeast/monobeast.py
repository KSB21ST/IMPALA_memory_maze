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

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import argparse
import logging
logging.basicConfig(level=logging.INFO)
import pprint
import threading
import time
import timeit
import traceback
import typing
import gym
import numpy as np
import wandb
# from memorymaze.gui.recording import SaveNpzWrapper
import memory_maze
from einops import rearrange, repeat, reduce
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

os.environ["OMP_NUM_THREADS"] = "1"  # Necessary for multithreading.

# if 'MUJOCO_GL' not in os.environ:
#     os.environ['MUJOCO_GL'] = 'egl'

import torch
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from torchbeast import atari_wrappers
from torchbeast.core import environment
from torchbeast.core import file_writer
from torchbeast.core import prof
from torchbeast.core import vtrace
from torchbeast.model import AtariNet

# yapf: disable
parser = argparse.ArgumentParser(description="PyTorch Scalable Agent")

# parser.add_argument("--env", type=str, default="PongNoFrameskip-v4",
#                     help="Gym environment.")
parser.add_argument("--env", type=str, default="memory_maze:MemoryMaze-9x9-ExtraObs-v0",
                    help="Gym environment.")
parser.add_argument("--mode", default="train",
                    choices=["train", "test", "test_render"],
                    help="Training or test mode.")
parser.add_argument("--wandb_mode", default="disabled",
                    choices=["online", "offline", "disabled"],
                    help="Training or test mode.")
parser.add_argument("--xpid", default=None,
                    help="Experiment id (default: None).")

# Training settings.
parser.add_argument("--disable_checkpoint", action="store_true",
                    help="Disable saving checkpoint.")
parser.add_argument("--savedir", default="~/logs/memory_maze",
                    help="Root dir where experiment data will be saved.")
parser.add_argument("--num_actors", default=4, type=int, metavar="N",
                    help="Number of actors (default: 4).")
parser.add_argument("--total_steps", default=100000000, type=int, metavar="T",
                    help="Total environment steps to train for.")
parser.add_argument("--batch_size", default=8, type=int, metavar="B",
                    help="Learner batch size.")
parser.add_argument("--unroll_length", default=80, type=int, metavar="T",
                    help="The unroll length (time dimension).")
parser.add_argument("--num_buffers", default=None, type=int,
                    metavar="N", help="Number of shared-memory buffers.")
parser.add_argument("--num_learner_threads", "--num_threads", default=2, type=int,
                    metavar="N", help="Number learner threads.")
parser.add_argument("--disable_cuda", action="store_true",
                    help="Disable CUDA.")
parser.add_argument("--gpu_n", type=int, default=0,
                    help="GPU #")
parser.add_argument("--use_lstm", action="store_true",
                    help="Use LSTM in agent model.")
parser.add_argument("--posemb", default="noemb",
                    help="type of position embedding: gt, noisygt, noemb")

# Loss settings.
parser.add_argument("--entropy_cost", default=0.0006,
                    type=float, help="Entropy cost/multiplier.")
parser.add_argument("--baseline_cost", default=0.5,
                    type=float, help="Baseline cost/multiplier.")
parser.add_argument("--discounting", default=0.99,
                    type=float, help="Discounting factor.")
parser.add_argument("--reward_clipping", default="abs_one",
                    choices=["abs_one", "none"],
                    help="Reward clipping.")

# Optimizer settings.
parser.add_argument("--learning_rate", default=0.00048,
                    type=float, metavar="LR", help="Learning rate.")
parser.add_argument("--alpha", default=0.99, type=float,
                    help="RMSProp smoothing constant.")
parser.add_argument("--momentum", default=0, type=float,
                    help="RMSProp momentum.")
parser.add_argument("--epsilon", default=0.01, type=float,
                    help="RMSProp epsilon.")
parser.add_argument("--grad_norm_clipping", default=40.0, type=float,
                    help="Global gradient norm clip.")
# yapf: enable


logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)

Buffers = typing.Dict[str, typing.List[torch.Tensor]]


import skvideo.io
import matplotlib.pyplot as plt

# viridis color list
colors = [
    [253, 231, 37],
    [241, 229, 29],
    [229, 228, 25],
    [216, 226, 25],
    [202, 225, 31],
    [186, 222, 40],
    [173, 220, 48],
    [160, 218, 57],
    [147, 215, 65],
    [132, 212, 75],
    [119, 209, 83],
    [108, 205, 90],
    [96, 202, 96],
    [86, 198, 103],
    [74, 193, 109],
    [64, 189, 114],
    [56, 185, 119],
    [49, 181, 123],
    [41, 175, 127],
    [37, 171, 130],
    [33, 166, 133],
    [31, 161, 135],
    [30, 157, 137],
    [31, 151, 139],
    [32, 146, 140],
    [33, 142, 141],
    [35, 137, 142],
    [37, 131, 142],
    [39, 127, 142],
    [41, 122, 142],
    [43, 117, 142],
    [45, 113, 142],
    [47, 107, 142],
    [49, 102, 142],
    [52, 97, 141],
    [54, 92, 141],
    [57, 85, 140],
    [60, 80, 139],
    [62, 74, 137],
    [64, 69, 136],
    [66, 63, 133],
    [69, 56, 130],
    [70, 50, 126],
    [71, 44, 122],
    [72, 37, 118],
    [72, 29, 111],
    [72, 23, 105],
    [71, 16, 99],
    [70, 8, 92],
    [68, 1, 84],
]  

def save_video(video, dir):
    fps = '20'
    crf = '17'
    vid_out = skvideo.io.FFmpegWriter(f'{dir}.mp4', 
                inputdict={'-r': fps},
                outputdict={'-r': fps, '-c:v': 'libx264', '-crf': crf, 
                            '-preset': 'ultrafast', '-pix_fmt': 'yuv444p'}
    )

    for frame in video:
        vid_out.writeFrame(frame)
    vid_out.close()
    

def generate_video(image, maze_layout, agent_pos, agent_dir, top_camera):
    
    print(image.shape, top_camera.shape, agent_pos.shape)
    l, bx, by = maze_layout.shape
    _, h, w, c = image.shape
    assert h==w and (h==256 or h==64)
    assert bx==by
    agent_pose = np.concatenate((agent_pos, agent_dir), axis=1)  # [l,4]
    margin = 32 if h == 256 else 8
    h_fig = h - margin * 2  # actual top camera size

    # draw pose on topdown
    dir_len = 20 if h == 256 else 10
    for j in range(l):
        
        # draw current pose
        for i in range(j+1):
            x = round(agent_pos[i, 0] * h_fig / bx) + margin
            y = h - round(agent_pos[i, 1] * h_fig / by) - margin
            
            if i == j:  # current pose with direction
                for n in range(dir_len):
                    x_ = x + round(agent_dir[i, 0] * n)
                    y_ = y - round(agent_dir[i, 1] * n)
                    if x_ < h and x_ > 0 and y_ < h and y_ > 0:
                        top_camera[j, y_, x_] = [0, 0, 0]
                            
            # top_camera[j, y, x] = [255, 0, 0]

    # video = np.concatenate((topdown, scenes), axis=2)
    video = np.concatenate((top_camera, image), axis=2)
    return video


def compute_baseline_loss(advantages):
    return 0.5 * torch.sum(advantages ** 2)


def compute_entropy_loss(logits):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    return torch.sum(policy * log_policy)


def compute_policy_gradient_loss(logits, actions, advantages):
    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1),
        reduction="none",
    )
    cross_entropy = cross_entropy.view_as(advantages)
    return torch.sum(cross_entropy * advantages.detach())


def act(
    flags,
    actor_index: int,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    model: torch.nn.Module,
    buffers: Buffers,
    initial_agent_state_buffers,
):
    try:
        logging.info("Actor %i started.", actor_index)
        timings = prof.Timings()  # Keep track of how fast things are.

        gym_env = create_env(flags)
        # seed = actor_index ^ int.from_bytes(os.urandom(4), byteorder="little")
        gym_env.seed(actor_index)
        env = environment.Environment(gym_env)
        env_output = env.initial()
        agent_state = model.initial_state(batch_size=1)
        agent_output, unused_state = model(env_output, agent_state)
        while True:
            index = free_queue.get()
            if index is None:
                break

            # Write old rollout end.
            for key in env_output:
                buffers[key][index][0, ...] = env_output[key]
            for key in agent_output:
                buffers[key][index][0, ...] = agent_output[key]
            for i, tensor in enumerate(agent_state):
                initial_agent_state_buffers[index][i][...] = tensor

            # Do new rollout.
            for t in range(flags.unroll_length):
                timings.reset()

                with torch.no_grad():
                    agent_output, agent_state = model(env_output, agent_state)

                timings.time("model")

                env_output = env.step(agent_output["action"])

                timings.time("step")

                for key in env_output:
                    buffers[key][index][t + 1, ...] = env_output[key]
                for key in agent_output:
                    buffers[key][index][t + 1, ...] = agent_output[key]

                timings.time("write")
            full_queue.put(index)

        if actor_index == 0:
            logging.info("Actor %i: %s", actor_index, timings.summary())

    except KeyboardInterrupt:
        pass  # Return silently.
    except Exception as e:
        logging.error("Exception in worker process %i", actor_index)
        traceback.print_exc()
        print()
        raise e


def get_batch(
    flags,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    buffers: Buffers,
    initial_agent_state_buffers,
    timings,
    lock=threading.Lock(),
):
    with lock:
        timings.time("lock")
        indices = [full_queue.get() for _ in range(flags.batch_size)]
        timings.time("dequeue")
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1) for key in buffers
    }
    initial_agent_state = (
        torch.cat(ts, dim=1)
        for ts in zip(*[initial_agent_state_buffers[m] for m in indices])
    )
    timings.time("batch")
    for m in indices:
        free_queue.put(m)
    timings.time("enqueue")
    batch = {k: t.to(device=flags.device, non_blocking=True) for k, t in batch.items()}
    initial_agent_state = tuple(
        t.to(device=flags.device, non_blocking=True) for t in initial_agent_state
    )
    timings.time("device")
    return batch, initial_agent_state


def learn(
    flags,
    actor_model,
    model,
    batch,
    initial_agent_state,
    optimizer,
    scheduler,
    lock=threading.Lock(),  # noqa: B008
):
    """Performs a learning (optimization) step."""
    with lock:
        learner_outputs, unused_state = model(batch, initial_agent_state)

        # Take final value function slice for bootstrapping.
        bootstrap_value = learner_outputs["baseline"][-1]

        # Move from obs[t] -> action[t] to action[t] -> obs[t].
        batch = {key: tensor[1:] for key, tensor in batch.items()}
        learner_outputs = {key: tensor[:-1] for key, tensor in learner_outputs.items()}

        rewards = batch["reward"]
        if flags.reward_clipping == "abs_one":
            clipped_rewards = torch.clamp(rewards, -1, 1)
        elif flags.reward_clipping == "none":
            clipped_rewards = rewards

        discounts = (~batch["done"]).float() * flags.discounting

        vtrace_returns = vtrace.from_logits(
            behavior_policy_logits=batch["policy_logits"],
            target_policy_logits=learner_outputs["policy_logits"],
            actions=batch["action"],
            discounts=discounts,
            rewards=clipped_rewards,
            values=learner_outputs["baseline"],
            bootstrap_value=bootstrap_value,
        )

        pg_loss = compute_policy_gradient_loss(
            learner_outputs["policy_logits"],
            batch["action"],
            vtrace_returns.pg_advantages,
        )
        baseline_loss = flags.baseline_cost * compute_baseline_loss(
            vtrace_returns.vs - learner_outputs["baseline"]
        )
        entropy_loss = flags.entropy_cost * compute_entropy_loss(
            learner_outputs["policy_logits"]
        )

        total_loss = pg_loss + baseline_loss + entropy_loss

        episode_returns = batch["episode_return"][batch["done"]]
        stats = {
            "episode_returns": tuple(episode_returns.cpu().numpy()),
            "mean_episode_return": torch.mean(episode_returns).item(),
            "total_loss": total_loss.item(),
            "pg_loss": pg_loss.item(),
            "baseline_loss": baseline_loss.item(),
            "entropy_loss": entropy_loss.item(),
        }

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), flags.grad_norm_clipping)
        optimizer.step()
        scheduler.step()

        actor_model.load_state_dict(model.state_dict())
        return stats


def create_buffers(flags, obs_shape, num_actions) -> Buffers:
    T = flags.unroll_length
    specs = dict(
        frame=dict(size=(T + 1, *obs_shape), dtype=torch.uint8),
        reward=dict(size=(T + 1,), dtype=torch.float32),
        done=dict(size=(T + 1,), dtype=torch.bool),
        episode_return=dict(size=(T + 1,), dtype=torch.float32),
        episode_step=dict(size=(T + 1,), dtype=torch.int32),
        policy_logits=dict(size=(T + 1, num_actions), dtype=torch.float32),
        baseline=dict(size=(T + 1,), dtype=torch.float32),
        last_action=dict(size=(T + 1,), dtype=torch.int64),
        action=dict(size=(T + 1,), dtype=torch.int64),
        # additional infos
        agent_pos=dict(size=(T + 1, 2), dtype=torch.float32),
        agent_dir=dict(size=(T + 1, 2), dtype=torch.float32),
        target_pos=dict(size=(T + 1, 2), dtype=torch.float32),
    )
    buffers: Buffers = {key: [] for key in specs}
    for _ in range(flags.num_buffers):
        for key in buffers:
            buffers[key].append(torch.empty(**specs[key]).share_memory_())
    return buffers


def train(flags):  # pylint: disable=too-many-branches, too-many-statements
    
    assert flags.posemb in ['noemb', 'gt', 'noisygt']
    if flags.xpid is None:
        flags.xpid = "%s/torchbeast-%s" % (flags.posemb, time.strftime("%Y%m%d-%H%M%S"))
    plogger = file_writer.FileWriter(
        xpid=flags.xpid, xp_args=flags.__dict__, rootdir=flags.savedir
    )
    checkpointpath = os.path.expandvars(
        os.path.expanduser("%s/%s/" % (flags.savedir, flags.xpid))
    )

    # wandb save
    wandb.login()
    wandb.init(
        project=f"impala_memmaze",
        entity="junmokane",
        config=flags,
        mode=flags.wandb_mode,
        name=flags.xpid,
    )

    if flags.num_buffers is None:  # Set sensible default for num_buffers.
        flags.num_buffers = max(2 * flags.num_actors, flags.batch_size)
    if flags.num_actors >= flags.num_buffers:
        raise ValueError("num_buffers should be larger than num_actors")
    if flags.num_buffers < flags.batch_size:
        raise ValueError("num_buffers should be larger than batch_size")

    T = flags.unroll_length
    B = flags.batch_size

    flags.device = None
    if not flags.disable_cuda and torch.cuda.is_available():
        logging.info("Using CUDA.")
        flags.device = torch.device(f"cuda:{flags.gpu_n}")
        torch.cuda.set_device(flags.device)
        # flags.device = torch.device(f"cuda")
    else:
        logging.info("Not using CUDA.")
        flags.device = torch.device("cpu")

    env = create_env(flags)

    print("$"*100)
    print(env.observation_space.shape, env.action_space.n)

    model = Net(env.observation_space.shape, env.action_space.n, flags.posemb, flags.use_lstm)
    eval_model = Net(env.observation_space.shape, env.action_space.n, flags.posemb, flags.use_lstm)
    buffers = create_buffers(flags, env.observation_space.shape, model.num_actions)

    model.share_memory()

    # Add initial RNN state.
    initial_agent_state_buffers = []
    for _ in range(flags.num_buffers):
        state = model.initial_state(batch_size=1)
        for t in state:
            t.share_memory_()
        initial_agent_state_buffers.append(state)

    actor_processes = []
    # ctx = mp.get_context("fork")
    ctx = mp.get_context("spawn")
    free_queue = ctx.SimpleQueue()
    full_queue = ctx.SimpleQueue()

    # act(flags, 0, free_queue, full_queue, model, buffers, initial_agent_state_buffers,)
    # exit()

    for i in range(flags.num_actors):
        actor = ctx.Process(
            target=act,
            args=(
                flags,
                i,
                free_queue,
                full_queue,
                model,
                buffers,
                initial_agent_state_buffers,
            ),
        )
        actor.start()
        actor_processes.append(actor)

    learner_model = Net(
        env.observation_space.shape, env.action_space.n, flags.posemb, flags.use_lstm
    ).to(device=flags.device)

    optimizer = torch.optim.Adam(
        learner_model.parameters(),
        lr=flags.learning_rate,
        eps=flags.epsilon,
    )

    def lr_lambda(epoch):
        return 1 - min(epoch * T * B, flags.total_steps) / flags.total_steps

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    logger = logging.getLogger("logfile")
    stat_keys = [
        "total_loss",
        "mean_episode_return",
        "pg_loss",
        "baseline_loss",
        "entropy_loss",
    ]
    logger.info("# Step\t%s", "\t".join(stat_keys))

    step, stats = 0, {}

    def batch_and_learn(i, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal step, stats
        timings = prof.Timings()
        while step < flags.total_steps:
            timings.reset()
            batch, agent_state = get_batch(
                flags,
                free_queue,
                full_queue,
                buffers,
                initial_agent_state_buffers,
                timings,
            )

            '''
            frame torch.Size([101, 32, 3, 64, 64])
            reward torch.Size([101, 32])
            done torch.Size([101, 32])
            episode_return torch.Size([101, 32])
            episode_step torch.Size([101, 32])
            policy_logits torch.Size([101, 32, 6])
            baseline torch.Size([101, 32])
            last_action torch.Size([101, 32])
            action torch.Size([101, 32])
            agent_pos torch.Size([101, 32, 2])
            agent_dir torch.Size([101, 32, 2])
            target_pos torch.Size([101, 32, 2])
            '''

            stats = learn(
                flags, model, learner_model, batch, agent_state, optimizer, scheduler
            )
            timings.time("learn")
            with lock:
                to_log = dict(step=step)
                to_log.update({k: stats[k] for k in stat_keys})
                plogger.log(to_log)
                # wandb.log(to_log, step=step)
                step += T * B

        if i == 0:
            logging.info("Batch and learn: %s", timings.summary())

    for m in range(flags.num_buffers):
        free_queue.put(m)

    threads = []
    for i in range(flags.num_learner_threads):
        thread = threading.Thread(
            target=batch_and_learn, name="batch-and-learn-%d" % i, args=(i,)
        )
        thread.start()
        threads.append(thread)

    def checkpoint(step):

        import math
        millnames = ['','K','M','B']
        def millify(n):
            n = float(n)
            millidx = max(0,min(len(millnames)-1, int(math.floor(0 if n == 0 else math.log10(abs(n))/3))))
            return '{:.0f}{}'.format(n / 10**(3 * millidx), millnames[millidx])

        ckpt_path = f'{checkpointpath}model_{millify(step)}.tar'
        if flags.disable_checkpoint:
            return
        logging.info("Saving checkpoint to %s", ckpt_path)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "flags": vars(flags),
            },
            ckpt_path,
        )

    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer()
        counter = -1
        while step < flags.total_steps:
            start_step = step
            start_time = timer()
            time.sleep(5)

            if timer() - last_checkpoint_time > 120 * 60:  # Save every 120 min.
                checkpoint(step)
                last_checkpoint_time = timer()

            sps = (step - start_step) / (timer() - start_time)
            if stats.get("episode_returns", None):
                mean_return = (
                    "Return per episode: %.1f. " % stats["mean_episode_return"]
                )

                wandb.log({'mean_episode_return_train': stats["mean_episode_return"],
                           'baseline_loss': stats['baseline_loss'],
                           'entropy_loss': stats['entropy_loss'],
                           'pg_loss': stats['pg_loss'],
                           'total_loss': stats['total_loss']},
                           step=step)
            else:
                mean_return = ""
            total_loss = stats.get("total_loss", float("inf"))
            logging.info(
                "Steps %i @ %.1f SPS. Loss %f. %sStats:\n%s",
                step,
                sps,
                total_loss,
                mean_return,
                pprint.pformat(stats),
            )

            '''# performance in eval mode'''
            if (step // 250000) > counter:
                counter = (step // 250000)
                num_episodes = 5
                # load learner_model to eval_model
                eval_model.load_state_dict(learner_model.state_dict())
                eval_model.eval()
                gym_env = create_env(flags)
                env = environment.Environment(gym_env)
                observation = env.initial()
                state = model.initial_state(batch_size=1)
                returns = [] 

                while len(returns) < num_episodes:
                    agent_outputs = eval_model(observation, state)
                    policy_outputs, state = agent_outputs
                    observation = env.step(policy_outputs["action"])
                    if observation["done"].item():
                        returns.append(observation["episode_return"].item())
                        logging.info(
                            "Episode ended after %d steps. Return: %.1f",
                            observation["episode_step"].item(),
                            observation["episode_return"].item(),
                        )
                env.close()
                wandb.log({'mean_episode_return_eval': sum(returns) / len(returns)}, step=counter*250000)
                logging.info("Average returns over %i steps: %.1f", num_episodes, sum(returns) / len(returns))

    except KeyboardInterrupt:
        return  # Try joining actors then quit.
    else:
        for thread in threads:
            thread.join()
        logging.info("Learning finished after %d steps.", step)
    finally:
        for _ in range(flags.num_actors):
            free_queue.put(None)
        for actor in actor_processes:
            actor.join(timeout=1)

    checkpoint(step)
    plogger.close()


def test(flags, num_episodes: int = 100):
    if flags.xpid is None:
        checkpointpath = "../../logs/memory_maze/torchbeast-20230428-190628/model.tar"
    else:
        checkpointpath = os.path.expandvars(
            os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "model.tar"))
        )
    # flags.device = torch.device(f"cuda:{flags.gpu_n}")
    # torch.cuda.set_device(flags.device)
    flags.device = torch.device(f"cuda")

    gym_env = create_env(flags)
    env = environment.Environment(gym_env)
    model = Net((3, 64,64), gym_env.action_space.n, flags.posemb, flags.use_lstm)
    # .to(device=flags.device)
    model.eval()
    checkpoint = torch.load(checkpointpath, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    observation = env.initial()
    resize = transforms.Resize(64)
    observation['frame'] = rearrange(resize(observation['frame'][0, 0]), 'c h w -> 1 1 c h w')

    # observation = {k: t.to(device=flags.device, non_blocking=True) for k, t in observation.items()}
    returns = []
    state = model.initial_state(batch_size=1)
    ep = {'image': [], 'agent_pos': [], 'agent_dir': [], 'top_camera': [], 'maze_layout': []}

    for k, v in ep.items():
        v.append(observation[k])  

    while len(returns) < num_episodes:
        if flags.mode == "test_render":
            env.gym_env.render()
        agent_outputs = model(observation, state)
        policy_outputs, state = agent_outputs
        observation = env.step(policy_outputs["action"])
        observation['frame'] = rearrange(resize(observation['frame'][0, 0]), 'c h w -> 1 1 c h w')
        # observation['frame'] = resize(observation['frame'])

        for k, v in ep.items():
            v.append(observation[k])
        # observation = {k: t.to(device=flags.device, non_blocking=True) for k, t in observation.items()}
        if observation["done"].item():
            returns.append(observation["episode_return"].item())
            logging.info(
                "Episode ended after %d steps. Return: %.1f",
                observation["episode_step"].item(),
                observation["episode_return"].item(),
            )
            for k, v in ep.items():
                ep[k] = np.array(v)
            ep['image'] = rearrange(ep['image'], 'l c h w -> l h w c')
            video = generate_video(image=ep['image'], maze_layout=ep['maze_layout'], agent_pos=ep['agent_pos'], agent_dir=ep['agent_dir'], top_camera=ep['top_camera'])
            save_video(video, dir=f'./traj_final256{len(returns)}_{observation["episode_return"].item()}')
            ep = {'image': [], 'agent_pos': [], 'agent_dir': [], 'top_camera': [], 'maze_layout': []}

    env.close()

    print(sum(returns) / len(returns))

    logging.info(
            "Average returns over %i steps: %.1f", num_episodes, sum(returns) / len(returns)
        )


Net = AtariNet

def create_env(flags):
    if flags.mode == "train":
        env = gym.make(flags.env)
        return atari_wrappers.wrap_pytorch(env)
    else:
        # env = gym.make('MemoryMaze-Custom-v0')
        env = gym.make(flags.env)
        return atari_wrappers.wrap_pytorch_test(env)
    

def main(flags):
    if flags.mode == "train":
        train(flags)
    else:
        test(flags)


if __name__ == "__main__":
    flags = parser.parse_args()
    main(flags)