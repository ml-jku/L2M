import os
import hydra
import wandb
import omegaconf
import numpy as np
from pathlib import Path
from stable_baselines3 import SAC, PPO, TD3, DQN
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape, is_image_space
from transformers import DecisionTransformerConfig
from torch.distributed import init_process_group, destroy_process_group
from src.algos import get_model_class, get_agent_class, AGENT_CLASSES, ContinualSAC
from src.envs import make_env
from src.callbacks import make_callbacks
from src.utils import maybe_split
from src.buffers import make_buffer_class


def setup_wandb(config):
    config_dict = omegaconf.OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    config_dict["PID"] = os.getpid()
    print(f"PID: {os.getpid()}")
    # hydra changes working directories automatically
    logdir = str(Path.joinpath(Path(os.getcwd()), config.logdir))
    Path(logdir).mkdir(exist_ok=True, parents=True)
    print(f"Logdir: {logdir}")

    run = None
    if config.use_wandb:
        print("Setting up logging to Weights & Biases.")
        # make "wandb" path, otherwise WSL might block writing to dir
        wandb_path = Path.joinpath(Path(logdir), "wandb")
        wandb_path.mkdir(exist_ok=True, parents=True)
        run = wandb.init(tags=[config.experiment_name],
                         config=config_dict, **config.wandb_params)
        print(f"Writing Weights & Biases logs to: {str(wandb_path)}")
        run.log_code(hydra.utils.get_original_cwd())
    return run, logdir


def setup_ddp():
    init_process_group(backend="nccl")


def make_agent(config, env, logdir):
    state_dim = get_obs_shape(env.observation_space)[0]
    act_dim = get_action_dim(env.action_space)
    agent_params_dict = omegaconf.OmegaConf.to_container(config.agent_params, resolve=True, throw_on_missing=True)
    agent_kind = agent_params_dict.pop("kind")
    agent_load_path = agent_params_dict.pop("load_path", None)
    agent_load_path = Path(agent_load_path["dir_path"]) / agent_load_path["file_name"] \
        if isinstance(agent_load_path, dict) else agent_load_path
    if agent_kind in AGENT_CLASSES.keys():
        if agent_kind in ["MDDT", "DDT", "MDMPDT"]: 
            # https://github.com/pytorch/pytorch/issues/11201#issuecomment-421146936
            import torch.multiprocessing
            torch.multiprocessing.set_sharing_strategy('file_system')
        
        # prespecified state/action dims in case of mixed spaces
        max_state_dim, max_act_dim = config.agent_params.replay_buffer_kwargs.get("max_state_dim", None), \
            config.agent_params.replay_buffer_kwargs.get("max_act_dim", None)
        if max_state_dim is not None: 
            state_dim = max_state_dim
        elif max_act_dim is not None:
            act_dim = max_act_dim
        
        # huggingface specific params
        agent_huggingface_params = agent_params_dict.pop("huggingface")
        dt_config = DecisionTransformerConfig(
            state_dim=state_dim,
            act_dim=act_dim,
            **agent_huggingface_params
        )

        # model specific params
        model_kwargs = agent_params_dict.pop("model_kwargs", {})
        if max_act_dim is not None:
            model_kwargs["max_act_dim"] = max_act_dim
            
        # exploration specific params
        action_noise_std = agent_params_dict.pop("action_noise_std", None)
        ou_noise = agent_params_dict.pop("ou_noise", False)
        if ou_noise:
            action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(act_dim),
                                                        sigma=action_noise_std * np.ones(act_dim)) \
                if action_noise_std is not None else None
        else:
            action_noise = NormalActionNoise(mean=np.zeros(act_dim), sigma=action_noise_std * np.ones(act_dim)) \
                if action_noise_std is not None else None

        # replay buffer class
        buffer_kind = agent_params_dict["replay_buffer_kwargs"].pop("kind", "default")
        replay_buffer_class = make_buffer_class(buffer_kind)
        
        # compose additional agent kwargs
        target_return = config.env_params.target_return
        reward_scale = config.env_params.reward_scale
        add_agent_kwargs = {
            "target_return": target_return / reward_scale if isinstance(reward_scale, (int, float)) else target_return,
            "reward_scale": reward_scale if isinstance(reward_scale, (int, float)) else dict(reward_scale),
            "device": config.device,
            "seed": config.seed,
            "tensorboard_log": logdir if config.use_wandb else None,
            "action_noise": action_noise,
            "load_path": agent_load_path,
            "replay_buffer_class": replay_buffer_class,
            "ddp": config.get("ddp", False)
        }

        # make DT model
        policy = get_model_class(agent_kind)(
            dt_config, env.observation_space, env.action_space,
            stochastic_policy=agent_params_dict["stochastic_policy"],
            **model_kwargs
        )
        # make DT agent
        agent = get_agent_class(agent_kind)(
            policy,
            env,
            **add_agent_kwargs,
            **agent_params_dict
        )
    elif agent_kind in ["SAC", "ContinualSAC"]:
        policy, policy_kwargs = agent_params_dict.pop("policy"), agent_params_dict.pop("policy_kwargs", {})
        extra_encoder = agent_params_dict.pop("extra_encoder")
        share_features_extractor = agent_params_dict.pop("share_features_extractor")
        features_extractor_arch = agent_params_dict.pop("features_extractor_arch")
        if extra_encoder:
            from src.algos.models.extractors import FlattenExtractorWithMLP
            policy_kwargs.update({"features_extractor_class": FlattenExtractorWithMLP,
                                  "share_features_extractor": share_features_extractor,
                                  "features_extractor_kwargs": {"net_arch": features_extractor_arch}})
        agent_class = ContinualSAC if agent_kind == "ContinualSAC" else SAC
        agent = agent_class(policy=policy,
                            env=env,
                            device=config.device,
                            seed=config.seed,
                            tensorboard_log=logdir if config.use_wandb else None,
                            verbose=1,
                            policy_kwargs=policy_kwargs,
                            **agent_params_dict)
        print(agent.policy)
    elif agent_kind == "TD3":
        policy = agent_params_dict.pop("policy")
        agent = TD3(policy=policy,
                    env=env,
                    device=config.device,
                    seed=config.seed,
                    tensorboard_log=logdir if config.use_wandb else None,
                    verbose=1,
                    action_noise=NormalActionNoise(mean=np.zeros(act_dim), sigma=0.1 * np.ones(act_dim)),
                    **agent_params_dict)
        print(agent.policy)
    elif agent_kind in ["PPO", "DQN"]:
        policy = agent_params_dict.pop("policy")
        agent_class = PPO if agent_kind == "PPO" else DQN
        agent = agent_class(policy=policy,
                            env=env,
                            device=config.device,
                            seed=config.seed,
                            tensorboard_log=logdir if config.use_wandb else None,
                            verbose=1,
                            **agent_params_dict)
        print(agent.policy)
    else:
        raise NotImplementedError
    return agent


@hydra.main(config_path="configs", config_name="config")
def main(config):
    print("Config: ", config)
    ddp = config.get("ddp", False)
    if ddp: 
        setup_ddp()
        # make sure only global rank0 writes to wandb
        logdir = None
        global_rank = int(os.environ["RANK"])
        if global_rank == 0: 
            run, logdir = setup_wandb(config)
    else: 
        run, logdir = setup_wandb(config)
    
    env, eval_env = make_env(config, logdir)
    agent = make_agent(config, env, logdir)
    callbacks = make_callbacks(config, eval_env=eval_env, logdir=logdir)
    res, score = None, None
    try:
        res = agent.learn(
            **config.run_params,
            eval_env=eval_env,
            callback=callbacks
        )
    finally:
        print("Finalizing run...")
        if config.use_wandb:
            if config.env_params.record:
                env.video_recorder.close()
            wandb.finish()
        # return last avg reward for hparam optimization
        score = None if res is None else safe_mean([ep_info["r"] for ep_info in res.ep_info_buffer])
        if ddp: 
            destroy_process_group()
    return score


if __name__ == "__main__":
    omegaconf.OmegaConf.register_new_resolver("maybe_split", maybe_split)
    main()
