# log_lrs=(-14 -13 -12 -11 -10)
# vres_list=(True False)
# for vres in "${vres_list[@]}"; do
#     for log_lr in "${log_lrs[@]}"; do
#         lr=$(python -c "print(2 ** $log_lr)")
#         torchrun --nnode=1 --nproc_per_node=8 train_gpt.py --run_name "layer96_sweep_gpt_${lr}_${vres}" --learning_rate $lr --vres $vres --n_layer 96
#     done
# done


log_embed_lrs=(-4 -6 -8 -10 -12)
log_lrs=(-14 -13 -12 -11 -10)

for log_lr in "${log_lrs[@]}"; do
    for log_embed_lr in "${log_embed_lrs[@]}"; do
        lr=$(python -c "print(2 ** $log_lr)")
        lr_embed=$(python -c "print(2 ** $log_embed_lr)")
        torchrun --nnode=1 --nproc_per_node=8 train_gpt.py --run_name "layer12_sweepembed_gpt_embed${lr_embed}_lr${lr}" --learning_rate $lr --learning_rate_embed $lr_embed --vres True --n_layer 12
    done
done

