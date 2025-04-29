export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.02

export EVAL_N_TRAJS=20
export CHECKPOINT_PATH=debug_rlif
export ENV_NAME=franka_sim

declare -A status
pids=()

for (( i = 2000; i <= 40000; i += 2000)); do
    /home/qirico/miniconda3/envs/hilserl3/bin/python ../../eval_sim_checkpoints.py \
        --exp_name=$ENV_NAME \
        --checkpoint_path=$CHECKPOINT_PATH \
        --eval_checkpoint_step=$i \
        --eval_n_trajs=$EVAL_N_TRAJS \
        >> log/$i.out 2>&1 &
    pids+=($!)
done

for pid in "${pids[@]}"; do
    if wait "$pid"; then
        status[$pid]=0
    else
        status[$pid]=$?
    fi
done

for pid in "${!status[@]}"; do
    printf "PID %-6 : exit code %s\n" "$pid" "${status[$pid]}"
done

echo "Everything done."
