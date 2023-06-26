#!/bin/bash
#SBATCH --account=X
#SBATCH --qos=X
#SBATCH --partition=X
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=2
#SBATCH --output=OUTPUT_PATH

source activate mddt

# on 2 GPUs (1 node, 2 GPUs per node) --> total batch size 1024
python main.py -m hydra/launcher=torchrun hydra.launcher.nproc_per_node=2 +ddp=True experiment_name=mddt_pretrain seed=42 env_params=multi_domain_mtdmc agent_params=cdt_pretrain_disc agent_params.kind=MDDT agent_params/model_kwargs=multi_domain_mtdmc agent_params/data_paths=mt40v2_dmc10 run_params=pretrain eval_params=pretrain_disc +agent_params/replay_buffer_kwargs=multi_domain_mtdmc +agent_params.accumulation_steps=2 +agent_params.use_amp=True +agent_params.batch_size=512
