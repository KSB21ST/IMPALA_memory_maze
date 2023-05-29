# Environment setup
Pleas refer to the public repository of [Memory maze](https://github.com/jurgisp/memory-maze)

Then set up our custom environment based on [Memory maze](https://github.com/jurgisp/memory-maze),

```shell
$ python steup.py install
```
## Gym Interface
You can create the environment using the [Gym](https://github.com/openai/gym) interface:

```python
!pip install gym
import gym

env = gym.make('memory_maze:MemoryMaze-9x9-v0')
env = gym.make('memory_maze:MemoryMaze-11x11-v0')
env = gym.make('memory_maze:MemoryMaze-13x13-v0')
env = gym.make('memory_maze:MemoryMaze-15x15-v0')
```

# TorchBeast
A PyTorch implementation of [IMPALA: Scalable Distributed
Deep-RL with Importance Weighted Actor-Learner Architectures
by Espeholt, Soyer, Munos et al.](https://arxiv.org/abs/1802.01561)

We used
[MonoBeast](#getting-started-monobeast) as our agent.

## Getting started: MonoBeast

MonoBeast is a pure Python + PyTorch implementation of IMPALA.

To set it up, create a new conda environment and install MonoBeast's
requirements:

```bash
$ conda create -n torchbeast
$ conda activate torchbeast
$ conda install pytorch -c pytorch
$ pip install -r requirements.txt
```

Then run MonoBeast, e.g. on the [Memory Maze](https://github.com/jurgisp/memory-maze):

```shell
$ python -m torchbeast.monobeast --env "memory_maze:MemoryMaze-15x15-ExtraObs-v0"
```

By default, MonoBeast uses only a few actors (each with their instance
of the environment). Let's change the default settings (try this on a
beefy machine!):

```shell
$ python -m torchbeast.monobeast \
    --env "memory_maze:MemoryMaze-15x15-ExtraObs-v0" \
    --posemb "noisygt" \
    --pos_noise 0.0 \
    --num_actors 12 \
    --discounting 0.99 \
    --entropy_cost 0.001 \
    --batch_size 32 \
    --unroll_length 100 \
    --learning_rate 2e-4 \
    --epsilon 1e-7 \
    --gpu_n 0 \
    --savedir "./logs"
```

Once training finished, we can test performance on a few episodes:

```shell
$ python -m torchbeast.monobeast \
    --posemb "noisygt" \
    --mode test \
    --record True \
    --env "memory_maze:MemoryMaze-15x15-ExtraObs-v0" \
    --pos_noise 0.0 \
    --num_actors 12 \
    --discounting 0.99 \
    --entropy_cost 0.001 \
    --batch_size 32 \
    --unroll_length 100 \
    --learning_rate 2e-4 \
    --epsilon 1e-7 \
    --gpu_n 0 \
```
you can also use run.sh as the following:
```shell
$ sh run.sh
```

## Repository contents

`libtorchbeast`: C++ library that allows efficient learner-actor
communication via queueing and batching mechanisms. Some functions are
exported to Python using pybind11. For PolyBeast only.

`nest`: C++ library that allows to manipulate complex
nested structures. Some functions are exported to Python using
pybind11.

`tests`: Collection of python tests.

`third_party`: Collection of third-party dependencies as Git
submodules. Includes [gRPC](https://grpc.io/).

`torchbeast`: Contains `monobeast.py`, and `polybeast.py`,
`polybeast_learner.py` and `polybeast_env.py`.


## Hyperparamaters

Both MonoBeast and PolyBeast have flags and hyperparameters. To
describe a few of them:

* `num_actors`: The number of actors (and environment instances). The
  optimal number of actors depends on the capabilities of the machine
  (e.g. you would not have 100 actors on your laptop). In default
  PolyBeast this should match the number of servers started.
* `batch_size`: Determines the size of the learner inputs.
* `unroll_length`: Length of a rollout (i.e., number of steps that an
  actor has to be perform before sending its experience to the
  learner). Note that every batch will have dimensions
  `[unroll_length, batch_size, ...]`.