log_lrs=(-13 -12 -11 -10 -9)
vres_list=(True False)
for vres in "${vres_list[@]}"; do
    for log_lr in "${log_lrs[@]}"; do
        lr=$(python -c "print(2 ** $log_lr)")
        torchrun --nnode=1 --nproc_per_node=8 train_gpt.py --run_name "layer48_sweep_gpt_${lr}_${vres}" --learning_rate $lr --vres $vres
    done
done
