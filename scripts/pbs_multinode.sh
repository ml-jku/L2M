#!/bin/bash -l
#PBS -A X
#PBS -q X
#PBS -l select=2
#PBS -j oe
#PBS -l walltime=48:00:00
#PBS -o OUTPUT_PATH

unset CUDA_VISIBLE_DEVICES
export NCCL_P2P_DISABLE=1

cd SOURCE_DIR_PATH
source activate mddt

# required for multinode
head_node=$(cat $PBS_NODEFILE | head -n 1)
head_node_ip=$(ssh $head_node hostname --ip-address)
RANDOM=84210

echo Head node: $head_node
echo Node IP: $head_node_ip
echo head_node_ip: $head_node_ip
export LOGLEVEL=INFO

# on 16 GPUs (2 nodes, 8 GPUs per node) --> total batch size 1024
python main.py -m hydra/launcher=torchrun +hydra.launcher.nnodes=2 hydra.launcher.nproc_per_node=8 hydra.launcher.max_nodes=2 hydra.launcher.rdzv_id=$RANDOM hydra.launcher.rdzv_backend=c10d hydra.launcher.rdzv_endpoint=$head_node_ip:29501 +ddp=True experiment_name=mddt_pretrain seed=42 env_params=multi_domain_mtdmc agent_params=cdt_pretrain_disc agent_params.kind=MDDT agent_params/model_kwargs=multi_domain_mtdmc agent_params/data_paths=mt40v2_dmc10 run_params=pretrain eval_params=pretrain_disc +agent_params/replay_buffer_kwargs=multi_domain_mtdmc +agent_params.accumulation_steps=2 +agent_params.use_amp=True +agent_params.batch_size=64