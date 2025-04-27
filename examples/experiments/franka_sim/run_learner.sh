export XLA_PYTHON_CLIENT_MEM_FRACTION=0.30
sudo /home/qirico/miniconda3/envs/hilserl3/bin/python ../../train_rlif.py "$@" \
    --exp_name=franka_sim \
    --checkpoint_path=debug_rlif \
    --demo_path=/home/qirico/Desktop/All-Weird/Human-Interventions/jax-hitl-hil-serl/examples/experiments/franka_sim/demo_data/franka_sim_1_demos_2025-04-26_17-37-28.pkl \
    --learner \
    --method=rlif \
