#!/bin/bash -l
#PBS -A X
#PBS -q X
#PBS -l select=1
#PBS -j oe
#PBS -l walltime=48:00:00
#PBS -o OUTPUT_PATH

unset CUDA_VISIBLE_DEVICES
cd SOURCE_DIR_PATH
source activate mddt

# on 8 GPUs (1 node, 8 GPUs per node) --> total batch size 1024
python main.py -m hydra/launcher=torchrun hydra.launcher.nproc_per_node=8 +ddp=True experiment_name=mddt_pretrain seed=42 env_params=multi_domain_mtdmc agent_params=cdt_pretrain_disc agent_params.kind=MDDT agent_params/model_kwargs=multi_domain_mtdmc agent_params/data_paths=mt40v2_dmc10 run_params=pretrain eval_params=pretrain_disc +agent_params/replay_buffer_kwargs=multi_domain_mtdmc +agent_params.accumulation_steps=2 +agent_params.use_amp=True +agent_params.batch_size=128