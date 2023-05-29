# For train
# python -m torchbeast.monobeast \
#     --env "memory_maze:MemoryMaze-15x15-ExtraObs-v0" \
#     --posemb "noisygt" \
#     --pos_noise 1.0 \
#     --num_actors 12 \
#     --discounting 0.99 \
#     --entropy_cost 0.001 \
#     --batch_size 32 \
#     --unroll_length 100 \
#     --learning_rate 2e-4 \
#     --epsilon 1e-7 \
#     --wandb_mode "online" \
#     --gpu_n 3 \
#     --savedir "../logs"

#for test
python -m torchbeast.monobeast \
    --posemb "noisygt" \
    --mode test \
    --record True \
    --env "memory_maze:MemoryMaze-15x15-ExtraObs-v0" \
    --pos_noise 0.5 \
    --num_actors 12 \
    --discounting 0.99 \
    --entropy_cost 0.001 \
    --batch_size 32 \
    --unroll_length 100 \
    --learning_rate 2e-4 \
    --epsilon 1e-7 \
    --wandb_mode "disabled" \
    --gpu_n 4 \
    --savedir "../logs"