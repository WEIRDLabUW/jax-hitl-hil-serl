export XLA_PYTHON_CLIENT_MEM_FRACTION=0.10
sudo /home/qirico/miniconda3/envs/hilserl3/bin/python ../../train_rlif.py "$@" \
    --exp_name=franka_sim \
    --checkpoint_path=debug_rlif \
    --actor \
    --method=rlif \
