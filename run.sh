# python -m torchbeast.monobeast \
#     --env "memory_maze:MemoryMaze-15x15-ExtraObs-v0" \
#     --posemb "noisygt" \
#     --pos_noise 0.0 \
#     --num_actors 16 \
#     --use_lstm \
#     --use_int_rew \
#     --discounting 0.99 \
#     --entropy_cost 0.001 \
#     --batch_size 32 \
#     --unroll_length 100 \
#     --learning_rate 2e-4 \
#     --epsilon 1e-7 \
#     --wandb_mode "disabled" \
#     --gpu_n 0 \
#     --savedir "/home/jmcho/project/IMPALA_memory_maze/logs"

python -m torchbeast.monobeast \
    --env "memory_maze:MemoryMaze-9x9-ExtraObs-v0" \
    --posemb "noisygt" \
    --pos_noise 0.05 \
    --num_actors 12 \
    --discounting 0.99 \
    --entropy_cost 0.001 \
    --batch_size 32 \
    --unroll_length 100 \
    --learning_rate 2e-4 \
    --epsilon 1e-7 \
    --wandb_mode "online" \
    --gpu_n 1 \
    --savedir "../logs"