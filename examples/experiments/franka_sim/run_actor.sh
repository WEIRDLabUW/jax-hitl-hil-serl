# XLA_PYTHON_CLIENT_MEM_FRACTION=0.10 /home/qirico/miniconda3/envs/hilserl3/bin/python ../../train_rlif.py "$@" \
#     --exp_name=franka_sim \
#     --checkpoint_path=debug_rlif_cta2_state \
#     --actor \
#     --method=rlif \

XLA_PYTHON_CLIENT_MEM_FRACTION=0.10 /home/qirico/miniconda3/envs/hilserl3/bin/python ../../train_rlif.py "$@" \
    --exp_name=franka_sim \
    --checkpoint_path=debug_soft_cl_cta2_state_fixed_3 \
    --actor \
    --method=soft_cl \
    --state_based \
