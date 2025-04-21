export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
sudo /home/qirico/miniconda3/envs/hilserl/bin/python ../../../train_rlif.py "$@" \
    --exp_name=franka_sim \
    --checkpoint_path=debug_rlif \
    --demo_path="" \
    --learner 
