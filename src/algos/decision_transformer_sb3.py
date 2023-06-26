import os
import time 
import collections
import numpy as np
import torch
import gym
from gym import spaces
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.utils import should_collect_more_steps, get_linear_fn, constant_fn
from stable_baselines3.common.type_aliases import RolloutReturn, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.logger import Figure

from .agent_utils import get_param_count, make_loss_fn, CustomDDP, make_random_proj_matrix
from ..buffers import TrajectoryReplayBuffer, PromptBuffer, ContinualTrajectoryReplayBuffer, MultiDomainTrajectoryReplayBuffer
from ..optimizers import make_optimizer
from ..envs.target_returns import ALL_TARGETS
from ..envs.env_names import ID_TO_DOMAIN
from ..envs.env_utils import extract_env_name
from ..utils.misc import make_promptcount_figures, make_attention_maps, make_sim_plot, set_frozen_to_eval
from ..schedulers import make_lr_scheduler
from ..augmentations import make_augmentations


class DecisionTransformerSb3(OffPolicyAlgorithm):

    def __init__(
            self,
            policy,
            env,
            learning_rate=1e-4,
            buffer_size: int = 10_000,
            learning_starts: int = 0,
            batch_size: int = 256,
            gamma: float = 1,
            train_freq=(1, "episode"),
            gradient_steps: int = 10,
            replay_buffer_class=TrajectoryReplayBuffer,
            replay_buffer_kwargs=None,
            tensorboard_log=None,
            verbose: int = 1,
            device="auto",
            create_eval_env: bool = False,
            seed=None,
            supported_action_spaces=None,
            max_grad_norm=.25,
            weight_decay=1e-2,
            target_return=3600,
            reward_scale=1000,
            ent_coef=0.0,
            lr_entropy=1e-4,
            ent_tuning_start=0,
            eval_context_len=5,
            offline_steps=0,
            warmup_steps=1000,
            accumulation_steps=1,
            target_return_mult=1,
            action_noise_prob=1.0,
            eval_deterministic_prob=0.0,
            exploration_fraction=0.0,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            pull_constraint_coef=0.5,
            label_smoothing=0.0,
            loss_fn="mse",
            target_entropy="auto",
            buffer_weight_by="len",
            buffer_max_len_type="trajectory",
            target_return_type="fixed",
            optimizer_kind="adamw",
            exploration_sched="constant",
            stochastic_policy=False,
            last_seq_only=False,
            use_prompt_buffer=False,
            rollout_eval_prompt=True,
            init_egreedy=True,
            log_attn_maps=False,
            persist_context=False,
            reset_optim_on_switch=False,
            frozen=False,
            set_frozen_eval=False,
            debug=False,
            use_amp=False,
            ddp=False,
            compile=False, 
            augment_params=None,
            buffer_reinit_percent=None,
            data_paths=None,
            prompt_data_paths=None,
            action_noise=None,
            target_reward=None,
            buffer_topk=None,
            buffer_target_return=None,
            reinit_policy_after=None,
            rollout_obs_noise_std=None,
            rollout_act_noise_std=None,
            max_grad_steps=None,
            steps_per_task=None,
            prompt_buffer_kwargs=None,
            prompt_buffer_sync_freq=None,
            eval_prompt_update_freq=None,
            load_path=None,
            load_kwargs=None,
            freeze_kwargs=None,
            lr_sched_kwargs=None
    ):
        super().__init__(
            policy,
            env,
            None,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            gamma=gamma,
            train_freq=tuple(train_freq) if isinstance(train_freq, list) else train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            sde_support=False,
            supported_action_spaces=supported_action_spaces,
            support_multi_env=True,
        )
        self.max_grad_norm = max_grad_norm
        self.weight_decay = weight_decay
        self.target_return = target_return
        self._reward_scale = reward_scale
        self.obs_shape = get_obs_shape(self.observation_space)
        self.action_dim = get_action_dim(self.action_space)
        self.stochastic_policy = stochastic_policy
        self.ent_coef = ent_coef
        self.target_entropy = target_entropy
        self.buffer_reinit_percent = buffer_reinit_percent
        self.lr_entropy = lr_entropy
        self.eval_context_len = eval_context_len
        self.data_paths = data_paths
        self.prompt_data_paths = prompt_data_paths
        self.offline_steps = offline_steps
        self.target_return_type = target_return_type
        self.target_return_mult = target_return_mult
        self.loss_fn_type = loss_fn
        self.label_smoothing = label_smoothing
        self.buffer_weight_by = buffer_weight_by
        self.warmup_steps = warmup_steps
        self.accumulation_steps = accumulation_steps
        self.target_reward = target_reward
        self.ent_tuning_start = ent_tuning_start
        self.augment_params = augment_params
        self.state_mean = None
        self.state_std = None
        self.optimizer_kind = optimizer_kind
        self.eval_deterministic_prob = eval_deterministic_prob
        self.action_noise_prob = action_noise_prob
        self.buffer_topk = buffer_topk
        self.buffer_target_return = buffer_target_return
        self.reinit_policy_after = reinit_policy_after
        self.buffer_max_len_type = buffer_max_len_type
        self.rollout_obs_noise_std = rollout_obs_noise_std
        self.rollout_act_noise_std = rollout_act_noise_std
        self.max_grad_steps = max_grad_steps
        self.last_seq_only = last_seq_only
        self.steps_per_task = steps_per_task
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.exploration_sched = exploration_sched
        self.exploration_rate = 0.0
        self.use_prompt_buffer = use_prompt_buffer
        self.prompt_buffer_kwargs = prompt_buffer_kwargs if prompt_buffer_kwargs is not None else {}
        self.prompt_buffer_sync_freq = 10000 if prompt_buffer_sync_freq is None else prompt_buffer_sync_freq
        self.eval_prompt_update_freq = 10000 if eval_prompt_update_freq is None else eval_prompt_update_freq
        self.eval_prompt = None
        self.rollout_eval_prompt = rollout_eval_prompt
        self.pull_constraint_coef = pull_constraint_coef
        self.lr_sched_kwargs = lr_sched_kwargs
        self.freeze_kwargs = freeze_kwargs
        self.load_kwargs = load_kwargs
        self.load_path = load_path
        self.frozen = frozen
        self.set_frozen_eval = set_frozen_eval
        self.init_egreedy = init_egreedy
        self.log_attn_maps = log_attn_maps
        self.persist_context = persist_context
        self.use_amp = use_amp      
        # self.amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) \
        #     else torch.float16
        self.ddp = ddp
        self.compile = compile 
        # DDP requires all forward() ouputs to be involved in loss computation. Disable some outputs if ddp is used
        self.ddp_kwargs = {"predict_reward": not ddp, "predict_return": not ddp, "predict_state": not ddp}
        self.reset_optim_on_switch = reset_optim_on_switch
        self.target_return_dict = collections.defaultdict(lambda: -float("inf"))
        # used to persist context during rollouts (necessary if performing updates during epoch)
        self.rollout_buffer = {}

        self._setup_model()
        # after setup model to ensure correct device in DDP
        self.current_task_id = 0
        self.current_task_id_tensor = torch.tensor(self.current_task_id, device=self.device, dtype=torch.int32)
        self.amp_dtype = torch.float16
        
        self.debug = debug
        if self.debug:
            from ..utils.debug import GradPlotter
            self.grad_plotter = GradPlotter()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        # set random seed does unnecessar device assert, change for DDP afterwards
        self.set_random_seed(self.seed)
        # turn off deterministic mode, as it slows down training considerably
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        if self.ddp: 
            self.device = int(os.environ["LOCAL_RANK"])
            self.global_rank = int(os.environ["RANK"])            
        self._setup_replay_buffer()
        self._setup_policy()
        self._setup_critic()
        if self.stochastic_policy:
            self._setup_entropy_tuning()
        self._setup_loss_fns()
        self._convert_train_freq()
        self._setup_exploration_schedule()
        self._setup_image_transforms()

    def _setup_replay_buffer(self):
        self.replay_buffer_kwargs.update({
            "device": self.device, "n_envs": self.n_envs,
            "target_return": self.buffer_target_return,
            "max_len_type": self.buffer_max_len_type, "last_seq_only": self.last_seq_only,
            "ddp": self.ddp, "augment_params": self.augment_params
        })
        if hasattr(self.policy_class, "config"):
            self.replay_buffer_kwargs["context_len"] = self.policy_class.config.max_length
            self.replay_buffer_kwargs["max_len"] = self.policy_class.config.max_ep_len

        self.replay_buffer = self.replay_buffer_class(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            **self.replay_buffer_kwargs,
        )
        if self.data_paths is not None:
            self.replay_buffer.init_buffer_from_dataset(self.data_paths)
        if self.use_prompt_buffer:
            self._setup_prompt_buffer()

    def _setup_prompt_buffer(self):
        # i.e., ensures that if prompt_buffer_kwargs are passed and cotain same args as in replay_buffer_kwargs
        # then the prompt buffer kwargs are used.
        buffer_kwargs = {**self.replay_buffer_kwargs}
        buffer_kwargs.update(self.prompt_buffer_kwargs)
        self.prompt_buffer = PromptBuffer(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            **buffer_kwargs
        )
        if self.prompt_data_paths:
            self.prompt_buffer.init_buffer_from_dataset(self.prompt_data_paths)

    def _setup_policy(self):
        self.policy = self.policy_class
        self.policy = self.policy.to(self.device)
        if torch.__version__ >= "2.0.0" and self.compile: 
            self.policy = torch.compile(self.policy)
        if self.load_path is not None:
            self.load_model_weights(self.load_path, freeze=self.frozen)    
        params = self.policy.get_optim_groups(weight_decay=self.weight_decay) if self.weight_decay > 0 \
            else self.policy.parameters()
        self.optimizer = make_optimizer(self.optimizer_kind, params, lr=self.learning_rate)
        load_kwargs = self.load_kwargs if self.load_kwargs is not None else {}
        load_optim = self.load_path is not None and load_kwargs.get("load_optim", False)
        if load_optim: 
            self.load_optim_state(self.load_path)
        
        if self.ddp: 
            torch.cuda.set_device(self.device)
            torch.cuda.empty_cache()
            self.policy = CustomDDP(self.policy, device_ids=[self.device], find_unused_parameters=True)
        
        # grad scaler for AMP
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # make schedulers
        schedulers, milestones = [], []
        if self.warmup_steps > 0 and not load_optim:
            # if loading optim, no warmup steps
            warmup_sched = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda steps: min((steps + 1) / self.warmup_steps, 1))
            schedulers.append(warmup_sched)
        if self.lr_sched_kwargs is not None:
            lr_sched_kind = self.lr_sched_kwargs.pop("kind")
            if load_optim: 
                # if loading warmup steps, continue cycle 
                self.lr_sched_kwargs["last_epoch"] = self.num_timesteps
            lr_sched = make_lr_scheduler(self.optimizer, kind=lr_sched_kind, sched_kwargs=self.lr_sched_kwargs)
            schedulers.append(lr_sched)
            milestones = [self.warmup_steps] if self.warmup_steps > 0 and not load_optim else []
        if len(schedulers) > 0: 
            self.schedulers = torch.optim.lr_scheduler.SequentialLR(self.optimizer, schedulers, milestones=milestones)
        else: 
            self.schedulers = None
        print(self.policy)
        print(self.optimizer)

    def _setup_critic(self):
        pass

    def _setup_exploration_schedule(self):
        if self.exploration_sched == "constant":
            self.exploration_schedule = constant_fn(self.exploration_fraction)
        else:
            self.exploration_schedule = get_linear_fn(
                self.exploration_initial_eps,
                self.exploration_final_eps,
                self.exploration_fraction,
            )

    def _setup_entropy_tuning(self):
        self._setup_target_entropy()
        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = torch.log(torch.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = make_optimizer(self.optimizer_kind, [self.log_ent_coef], lr=self.lr_entropy)

            if self.warmup_steps > 0:
                self.ent_coef_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.ent_coef_optimizer,
                    lambda steps: min((steps + 1) / self.warmup_steps, 1)
                )
            else:
                self.ent_coef_scheduler = None
        else:
            self.ent_coef_optimizer = None
            self.ent_coef_scheduler = None
            self.ent_coef_tensor = torch.tensor(float(self.ent_coef), device=self.device)

    def _setup_target_entropy(self):
        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
        else:
            self.target_entropy = float(self.target_entropy)

    def _setup_loss_fns(self, reduction="mean"):
        loss_fn = make_loss_fn(self.loss_fn_type, reduction=reduction, label_smoothing=self.label_smoothing)
        if loss_fn is not None:
            self.loss_fn = loss_fn.to(self.device)
            
    def _setup_image_transforms(self): 
        self.transforms = None
        if self.augment_params is not None: 
            self.transforms = make_augmentations(self.augment_params)
            print(self.transforms)
    
    @property
    def reward_scale(self):
        """
        Reward scale can either be a float or a dict mapping domain keys to floats. 
        When using accumulation steps, we assume that the current accumulation step maps to the index of the respective
        reward scale for the desired domain. However, to map the current accumulation step to the respective domain, we
        the replay buffer needs to maintain a counter of how many batches have been sampled with the current loader. 
        
        Returns: Float.   
        
        """
        if isinstance(self._reward_scale, dict):
            if self.steps_per_task is not None: 
                # CL case requires different handling. Assumes _reward_scale has a list 'cl'
                idx = min(self.current_task_id, len(self._reward_scale['cl']) - 1)
                return self._reward_scale['cl'][idx]
            divisor = self.accumulation_steps if self.accumulation_steps > 1 else len(self._reward_scale)
            idx = max(self.replay_buffer.num_sampled_batches - 1, 0) % divisor
            return list(self._reward_scale.values())[idx]
        return self._reward_scale
    
    def get_reward_scale_for_env(self, envid=None):
        """
        Returns the reward scale for the respective domain.
        When envid is given, we always return the reward scale for the respective domain.
        Args: 
            envid: Str. Id of the environment. Only used in case self._reward_scale is a dict.
        """
        if isinstance(self._reward_scale, dict) and envid is not None:
            return self._reward_scale[ID_TO_DOMAIN[envid]]
        return self._reward_scale

    @torch.no_grad()
    def collect_rollouts(
        self,
        env,
        callback,
        train_freq,
        replay_buffer,
        action_noise=None,
        learning_starts: int = 0,
        log_interval=None,
    ):
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.eval()
        # custom counter that is reset after every done
        current_timestep = 0
        action_stats = []
        num_collected_steps, num_collected_episodes = 0, 0
        assert train_freq.frequency > 0, "Should at least collect one step or episode."
        callback.on_rollout_start()
        continue_training = True

        # TODO: for now we assume that we have only one environment. Extend to parallel envs later
        # required for sequence model
        state = np.zeros((1, *self.obs_shape))
        self._last_obs = state
        target_return_val = self.compute_target_return_val(self.env, task_id=self.current_task_id)
        target_return_val_tensor = torch.as_tensor(target_return_val, device=self.device, dtype=torch.float32)
        if self.rollout_buffer == {}:
            target_return = torch.as_tensor(target_return_val, device=self.device, dtype=torch.float32).reshape(1, 1)
            states = torch.zeros((1, *self.obs_shape), device=self.device, dtype=torch.float32)
            actions = torch.zeros((0, self.action_dim), device=self.device, dtype=torch.float32)
            rewards = torch.zeros(0, device=self.device, dtype=torch.float32)
            timesteps = torch.as_tensor(0, device=self.device, dtype=torch.long).reshape(1, 1)
            if self.target_reward is not None and (self.policy.reward_condition or self.policy.reward_condition_only):
                target_reward = torch.as_tensor(self.target_reward / self.reward_scale, device=self.device,
                                                dtype=torch.float32).reshape(1, 1)
            else:
                target_reward = None
        else:
            states, actions, target_return, rewards, timesteps, target_reward, current_timestep = self.rollout_buffer.values()

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            actions = torch.cat([actions, torch.zeros((1, self.action_dim), device=self.device)], dim=0)
            rewards = torch.cat([rewards, torch.zeros(1, device=self.device)])
            action, buffer_action, action_tensor = self._sample_action(
                states, actions, target_reward if target_reward is not None else rewards,
                target_return, timesteps,
                learning_starts, action_noise, env.num_envs,
                context_len=self.eval_context_len,
                deterministic=np.random.rand() < self.eval_deterministic_prob,
                task_id=self.current_task_id_tensor
            )
            actions[-1] = action_tensor if action_tensor is not None else torch.from_numpy(action).to(actions.device)
            state, reward, done, info = env.step(action)
            action_stats.append(action)
            buffer_reward = reward.copy()
            self.num_timesteps += env.num_envs
            num_collected_steps += 1
            current_timestep += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(info, done)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_action, state, buffer_reward, done, info)
            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)
            self._on_step()

            # required for sequence model
            cur_state = torch.from_numpy(state).to(device=self.device).reshape(1, *self.obs_shape)
            states = torch.cat([states, cur_state], dim=0)
            buffer_reward = torch.from_numpy(buffer_reward).to(device=self.device)
            rewards[-1] = buffer_reward / self.reward_scale
            pred_return = torch.minimum(
                target_return[0, -1] - (buffer_reward / self.reward_scale), target_return_val_tensor
            )
            target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
            timesteps = torch.cat([timesteps, torch.ones((1, 1), device=self.device, dtype=torch.long) * (current_timestep)], dim=1)
            if target_reward is not None:
                self.target_reward = max(reward, self.target_reward)
                target_reward = torch.cat([target_reward,
                                           torch.as_tensor(self.target_reward / self.reward_scale, device=self.device).reshape(1, 1)], dim=1)
                target_reward = target_reward[-self.eval_context_len:]

            # prune sequence model buffers
            if len(states) > self.eval_context_len * 30:
                # avoids pruning at every timestep
                states = states[-self.eval_context_len:]
                actions = actions[-self.eval_context_len:]
                rewards = rewards[-self.eval_context_len:]
                target_return = target_return[:, -self.eval_context_len:]
                timesteps = timesteps[:, -self.eval_context_len:]

            for idx, d in enumerate(done):
                if d:
                    # Update stats
                    self._episode_num += 1
                    num_collected_episodes += 1
                    current_timestep = 0

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self.logger.record("rollout/action_mean", np.mean(action_stats))
                        self.logger.record("rollout/action_std", np.std(action_stats))
                        self._dump_logs()

                    if not self.persist_context:
                        # only reset if eval_context_len is shorter than episode length
                        # otherwise we want to have in-context learning and persist the context across episodes.
                        states = torch.zeros((1, *self.obs_shape), device=self.device, dtype=torch.float32)
                        target_return = torch.as_tensor(target_return_val, device=self.device, dtype=torch.float32).reshape(1, 1)
                        actions = torch.zeros((0, self.action_dim), device=self.device, dtype=torch.float32)
                        rewards = torch.zeros(0, device=self.device, dtype=torch.float32)
                        timesteps = torch.tensor(0, device=self.device, dtype=torch.long).reshape(1, 1)
                        if target_reward is not None:
                            target_reward = torch.as_tensor(self.target_reward / self.reward_scale, device=self.device,
                                                            dtype=torch.float32).reshape(1, 1)
                    if action_noise is not None:
                        action_noise.reset()

        callback.on_rollout_end()

        #  persist context during rollouts (necessary if performing updates during epoch and in-context RL)
        if not any(done) or self.persist_context:
            self.rollout_buffer = {
                "states": states[-self.eval_context_len:],
                "actions": actions[-self.eval_context_len:],
                "target_return": target_return[:, -self.eval_context_len:],
                "rewards": rewards[-self.eval_context_len:],
                "timesteps": timesteps[:, -self.eval_context_len:],
                "target_reward": target_reward,
                "current_timestep": current_timestep if any(done) else 0
            }
        else:
            self.rollout_buffer = {}

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def compute_target_return_val(self, env=None, task_id=0):
        if len(self.replay_buffer) > 0 and self.target_return_type == "max":
            if task_id == self.current_task_id:
                target = (self.replay_buffer._get_max_return() * self.target_return_mult) / self.reward_scale
                self.target_return_dict[task_id] = max(self.target_return_dict[task_id], target)
            else:
                # in case task_id already in return_dict, use it otherwise target_return=1
                target = self.target_return_dict[task_id] if task_id in self.target_return_dict else 1
            return target
        elif len(self.replay_buffer) > 0 and self.target_return_type == "mean_topk":
            return self.replay_buffer._get_mean_topk_return() / self.reward_scale
        elif len(self.replay_buffer) > 0 and self.target_return_type == "quantile":
            return self.replay_buffer._get_quantile_return() / self.reward_scale
        elif self.target_return_type == "predefined":
            env_name = extract_env_name(env, task_id)
            return (ALL_TARGETS[env_name] / self.get_reward_scale_for_env(env_name)) * self.target_return_mult
        return self.target_return

    def _sample_action(
        self,
        states,
        actions, rewards, target_return, timesteps,
        learning_starts: int,
        action_noise=None,
        n_envs: int = 1,
        context_len: int = 5,
        deterministic=False,
        task_id=None
    ):
        # action tensor avoids us having to move action once again to device
        action_tensor = None
        if ((self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup))
                or np.random.rand() < self.exploration_rate) and self.init_egreedy:
            # Warmup phase or e-greedy
            unscaled_action = self.action_space.sample()
            if isinstance(self.action_space, spaces.Discrete):
                unscaled_action = np.array(unscaled_action)
        else:
            prompt = None
            if self.use_prompt_buffer:
                if self.num_timesteps % self.eval_prompt_update_freq == 0 and self.rollout_eval_prompt:
                    self._update_eval_prompt()
                prompt = self.eval_prompt
            unscaled_action, _ = self.predict(
                self.policy, states, actions, rewards, target_return, timesteps,
                state=None, episode_start=None, deterministic=deterministic,
                context_len=context_len, prompt=prompt, task_id=task_id
            )

            action_tensor = unscaled_action
            unscaled_action = unscaled_action.detach().cpu().numpy()
            if action_noise is not None:
                if np.random.rand() < self.action_noise_prob:
                    noise = action_noise()
                    unscaled_action = np.clip(unscaled_action + noise, -1, 1)

        if isinstance(self.action_space, gym.spaces.Box):
            # necessary for us, as DT operates on one env by default
            unscaled_action = unscaled_action.reshape((n_envs, *unscaled_action.shape))
            scaled_action = self.policy.scale_action(unscaled_action)
            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
            if action_tensor is not None:
                action_tensor = action_tensor.reshape((n_envs, *unscaled_action.shape))
        else:
            # necessary for us, as DT operates on one env by default
            unscaled_action = unscaled_action.flatten()
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = np.copy(buffer_action)
            if action_tensor is not None:
                action_tensor = action_tensor.flatten()
        return action, buffer_action, action_tensor

    def predict(self, policy, observation, actions, rewards, returns_to_go, timesteps,
                state=None, episode_start=None, deterministic=True, context_len=5,
                prompt=None, task_id=None, is_eval=False, env_act_dim=None):
        if len(observation.shape) == 4 and self.transforms is not None:
            # is image 
            observation = self.transforms(observation)
        
        obs_shape, act_dim = observation.shape[1:], actions.shape[-1]
        states = observation.reshape(1, -1, *obs_shape)
        actions = actions.reshape(1, -1, act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        states = states[:, -context_len:]
        actions = actions[:, -context_len:]
        returns_to_go = returns_to_go[:, -context_len:]
        timesteps = timesteps[:, -context_len:]
        if rewards is not None:
            rewards = rewards.reshape(1, -1, 1)
            rewards = rewards[:, -context_len:]
        if self.state_mean is not None and self.state_std is not None:
            states = (states - self.state_mean) / self.state_std
        if self.rollout_obs_noise_std is not None and self.rollout_obs_noise_std > 0:
            noise = states.clone().data.normal_(0, self.rollout_obs_noise_std)
            states = (states.clone() + noise)
        if self.rollout_act_noise_std is not None and self.rollout_act_noise_std > 0:
            noise = actions.clone().data.normal_(0, self.rollout_act_noise_std)
            actions = (actions.clone() + noise)

        # pad all tokens to sequence length
        states, actions, returns_to_go, timesteps, attention_mask, rewards = self.pad_inputs(
            states, actions, returns_to_go, timesteps,
            context_len=context_len,
            # context_len=self.policy.config.max_length,
            rewards=rewards
        )

        # separate function, so that we can overwrite this in inheriting classes
        a1, a2 = self.get_action_pred(policy, states, actions, rewards, returns_to_go, timesteps,
                                      attention_mask, deterministic, prompt,
                                      task_id=task_id, is_eval=is_eval, env_act_dim=env_act_dim)
        return a1, a2

    def pad_inputs(self, states, actions, returns_to_go, timesteps, context_len=5, rewards=None):        
        padding = context_len - actions.shape[1]
        attention_mask = torch.cat([torch.zeros(padding, device=self.device, dtype=torch.long),
                                    torch.ones(actions.shape[1], device=self.device, dtype=torch.long)])
        attention_mask = attention_mask.reshape(1, -1)
        # TODO: use torch.nn.functional.pad instead
        if self.replay_buffer.max_state_dim is not None and len(states.shape) == 3:
            # pad state input to max_state_dim, in case of continous state
            s_pad = self.replay_buffer.max_state_dim - states.shape[-1]
            states = torch.cat([states, torch.zeros((*states.shape[:-1], s_pad), device=self.device)], dim=-1)
            # rand_proj_mat = torch.from_numpy(
            #     make_random_proj_matrix(states.shape[-1], self.replay_buffer.max_state_dim), 
            # ).to(self.device)
            # states = states.float() @ rand_proj_mat.T      
        if self.replay_buffer.max_act_dim is not None and actions.is_floating_point(): 
            # currently doesn't work, all actions are floating_point at this point
            # quick hack, check if observations are images --> discrete action (Atari)
            if len(states.shape) != 5: 
                a_pad = self.replay_buffer.max_act_dim - actions.shape[-1] 
                actions = torch.cat([actions, torch.zeros((*actions.shape[:-1], a_pad), device=self.device)], dim=-1) 
        obs_shape, act_dim = states.shape[2:], actions.shape[-1]
        states = torch.cat([torch.zeros((states.shape[0], padding, *obs_shape), device=self.device), states], dim=1).float()
        actions = torch.cat([torch.zeros((actions.shape[0], padding, act_dim), device=self.device), actions], dim=1)
        returns_to_go = torch.cat([torch.zeros((returns_to_go.shape[0], padding, 1), device=self.device), returns_to_go], dim=1).float()
        timesteps = torch.cat([torch.zeros((timesteps.shape[0], padding), dtype=torch.long, device=self.device), timesteps], dim=1)
        if rewards is not None:
            rewards = torch.cat([torch.zeros((rewards.shape[0], padding, 1), device=self.device), rewards], dim=1)
        return states, actions, returns_to_go, timesteps, attention_mask, rewards

    def get_action_pred(self, policy, states, actions, rewards, returns_to_go, timesteps, attention_mask,
                        deterministic, prompt,  is_eval=False, task_id=None, env_act_dim=None):
        with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
            policy_output = policy(
                states=states,
                actions=actions,
                rewards=rewards,
                returns_to_go=returns_to_go,
                timesteps=timesteps,
                attention_mask=attention_mask,
                return_dict=True,
                deterministic=deterministic,
                prompt=prompt,
                task_id=task_id,
                ddp_kwargs=self.ddp_kwargs
            )
        if not is_eval and self.num_timesteps % 10000 == 0 and self.log_attn_maps:
            self._record_attention_maps(policy_output.attentions, step=self.num_timesteps, prefix="rollout")
            if policy_output.cross_attentions is not None:
                self._record_attention_maps(policy_output.cross_attentions, step=self.num_timesteps,
                                            prefix="rollout_cross", lower_triu=False)
        action_preds = policy_output.action_preds
        return action_preds[0, -1], action_preds[0, -1]

    def train(self, gradient_steps: int, batch_size: int) -> None:
        self.policy.train()
        if self.set_frozen_eval:
            self.policy.apply(set_frozen_to_eval)
        metrics = collections.defaultdict(list)

        for _ in range(gradient_steps):
            observations, actions, next_observations, rewards, rewards_to_go, timesteps, attention_mask, \
                dones, task_ids, trj_ids, action_targets, action_mask, prompt = self.sample_batch(batch_size)

            # forward pass through policy
            with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                policy_output = self.policy(
                    states=observations,
                    actions=actions,
                    rewards=rewards,
                    returns_to_go=rewards_to_go,
                    timesteps=timesteps.long(),
                    attention_mask=attention_mask,
                    return_dict=True,
                    with_log_probs=self.stochastic_policy,
                    deterministic=False,
                    prompt=prompt,
                    task_id=self.current_task_id_tensor,
                    ddp_kwargs=self.ddp_kwargs,
                )
            action_log_probs, action_log_probs_masked, entropy_masked = None, None, None
            if self.stochastic_policy:
                action_log_probs = policy_output.action_log_probs
                action_log_probs_masked = action_log_probs.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
                metrics["action_log_probs_mean"].append(action_log_probs_masked.mean().item())
                if policy_output.entropy is not None:
                    entropy_masked = policy_output.entropy.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
                if self.last_seq_only:
                    # action_log_probs_masked is already masked. Only use last sequence for backward pass.
                    is_last_seq = torch.zeros_like(attention_mask)
                    is_last_seq[:, -1] = 1
                    is_last_seq = is_last_seq.reshape(-1)[attention_mask.reshape(-1) > 0] > 0
                    action_log_probs_masked = action_log_probs_masked[is_last_seq]
                    entropy_masked = entropy_masked[is_last_seq] if entropy_masked is not None else None

            # update ent_coef
            if self.stochastic_policy and self._n_updates > self.ent_tuning_start:
                ent_coef, ent_coef_dict = self.update_entropy_coef(action_log_probs_masked, entropy=entropy_masked)
                for k, v in ent_coef_dict.items():
                    metrics[k].append(v)
                ent_tuning = True
            else:
                ent_coef = 0
                ent_tuning = False

            # compute loss + update
            loss_dict = self.update_policy(
                policy_output, action_targets, attention_mask, ent_coef,
                return_targets=rewards_to_go, ent_tuning=ent_tuning,
                reward_targets=rewards, state_targets=observations, timesteps=timesteps, 
                dones=dones.float(), next_states=next_observations, action_mask=action_mask
            )
            for k, v in loss_dict.items():
                metrics[k].append(v)

            if (self._n_updates + 1) % 10000 == 0 and self.log_attn_maps:
                self._record_attention_maps(policy_output.attentions, step=self.num_timesteps, prefix="train")
                if policy_output.cross_attentions is not None:
                    self._record_attention_maps(policy_output.cross_attentions, step=self.num_timesteps,
                                                prefix="train_cross", lower_triu=False)
            metrics["target_returns"].append(rewards_to_go.mean().item())
            self._n_updates += 1

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self._record_metrics_from_dict(metrics, prefix="train")

    def sample_batch(self, batch_size):
        replay_data = self.replay_buffer.sample(
            batch_size=batch_size,
            weight_by=self.buffer_weight_by,
            env=self._vec_normalize_env,
            top_k=self.buffer_topk
        )
        observations, actions, next_observations, rewards, rewards_to_go, timesteps, \
            attention_mask, dones, task_ids, trj_ids, action_mask = replay_data
        action_targets = torch.clone(actions)

        if self.state_mean is not None and self.state_std is not None:
            observations = (observations - self.state_mean) / self.state_std
            next_observations = (next_observations - self.state_mean) / self.state_std
        if self.reward_scale != 1:
            rewards_to_go = rewards_to_go / self.reward_scale
            rewards = rewards / self.reward_scale

        # make prompt
        prompt = None
        if self.use_prompt_buffer:
            if not self.prompt_buffer.is_empty(self.current_task_id) or self.prompt_buffer.multitask_batch:
                prompt = self.prompt_buffer.sample(
                    batch_size=self.batch_size,
                    env=self._vec_normalize_env,
                    task_id=self.current_task_id if not self.prompt_buffer.multitask_batch else task_ids,
                    trj_id=trj_ids
                )
                prompt = prompt._replace(rewards_to_go=prompt.rewards_to_go / self.reward_scale,
                                         rewards=prompt.rewards / self.reward_scale)
        return observations, actions, next_observations, rewards, rewards_to_go, timesteps,\
            attention_mask, dones, task_ids, trj_ids, action_targets, action_mask, prompt

    def update_policy(self, policy_output, action_targets, attention_mask, ent_coef,
                      ent_tuning=True, return_targets=None, reward_targets=None, state_targets=None,
                      timesteps=None, dones=None, next_states=None, action_mask=None):
        with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
            loss, loss_dict = self.compute_policy_loss(
                policy_output, action_targets, attention_mask, ent_coef,
                ent_tuning=ent_tuning, return_targets=return_targets,
                reward_targets=reward_targets, state_targets=state_targets, dones=dones, 
                timesteps=timesteps, next_states=next_states, action_mask=action_mask
            )
            loss = loss / self.accumulation_steps
            
        # AMP scaling
        self.grad_scaler.scale(loss).backward()

        if self._n_updates % 100 == 0 and self.debug:
            if hasattr(self, "critic"):
                self.grad_plotter.plot_grad_flow(self.critic.named_parameters(), f"policy_update,critic,step={self._n_updates}.png")
            self.grad_plotter.plot_grad_flow(self.policy.named_parameters(), f"policy_update,policy,step={self._n_updates}.png")

        # AMP update step
        if self._n_updates % self.accumulation_steps == 0:
            # unscale grads:
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            if self.debug:
                params_before = {k: v.clone() for k, v in self.policy.named_parameters()}
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.optimizer.zero_grad(set_to_none=False)
            if self.policy.num_task_heads > 1:
                self.policy.zero_grads_for_action_heads(self.current_task_id)
            if self.debug:
                params_after = {k: v.clone() for k, v in self.policy.named_parameters()}
                has_changed = {k: not torch.equal(params_before[k], params_after[k]) for k in params_before}
                # print({k: v for k, v in has_changed.items() if v})
        if self.schedulers is not None: 
            self.schedulers.step()
            self.logger.record("train/learning_rate", self.schedulers.get_last_lr()[0])
        return loss_dict

    def compute_policy_loss(self, policy_output, action_targets, attention_mask, ent_coef,
                            ent_tuning=True, return_targets=None, reward_targets=None, state_targets=None, 
                            dones=None, timesteps=None, next_states=None, action_mask=None):
        action_preds, action_log_probs = policy_output.action_preds, policy_output.action_log_probs
        # compute loss + update
        loss_dict = {}
        act_dim = action_preds.shape[2]

        # shape: [batch_size, context_len, action_dim] (before masking)
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        if self.stochastic_policy:
            if self.loss_fn_type == "nll":
                # shape: [batch_size, context_len, action_dim] (before masking)
                action_targets = action_targets.reshape(-1, action_targets.shape[-1])
                action_targets_log_prob = self.policy.compute_log_prob_given_action(action_targets)
                loss_actions = -action_targets_log_prob.reshape(-1, 1)[attention_mask.reshape(-1) > 0].mean()
            else:
                # shape: [batch_size, context_len, action_dim] (before masking)
                action_targets = action_targets.reshape(-1, action_targets.shape[-1])[attention_mask.reshape(-1) > 0]
                loss_actions = self.loss_fn(action_preds, action_targets)
            if ent_tuning:
                # shape: [batch_size, context_len, 1] (before masking)
                action_log_probs = action_log_probs.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
                entropy = -torch.mean(action_log_probs)
                loss = loss_actions - (ent_coef * entropy)
                loss_dict["entropy"] = entropy.item()
            else:
                loss = loss_actions
            loss_dict["loss_actions"] = loss_actions.item()
        else:
            action_targets = action_targets.reshape(-1, action_targets.shape[-1])[attention_mask.reshape(-1) > 0]
            if self.policy.is_discrete:
                action_targets = action_targets.squeeze().long()
            loss = self.loss_fn(action_preds, action_targets)

        if hasattr(policy_output, "prompt_infos") and policy_output.prompt_infos is not None:
            loss_dict["loss_actions"] = loss.item()
            loss = self.compute_prompt_loss(loss, policy_output)

        loss_dict["loss"] = loss.item()
        return loss, loss_dict

    def compute_prompt_loss(self, loss, policy_output):
        prompt_infos = policy_output.prompt_infos
        if "reduce_sim" in prompt_infos:
            loss = loss - (self.pull_constraint_coef * policy_output.prompt_infos['reduce_sim'])
        self._record_prompt_infos(prompt_infos=policy_output.prompt_infos)
        return loss

    def update_entropy_coef(self, action_log_probs, entropy=None):
        ent_coef_dict = {}
        if self.ent_coef_optimizer is not None and self.stochastic_policy:
            # Important: detach the variable from the graph
            # so we don't change it with other losses
            # see https://github.com/rail-berkeley/softlearning/issues/60
            ent_coef = torch.exp(self.log_ent_coef.detach())
            if entropy is not None:
                ent_coef_loss = (self.log_ent_coef * (entropy - self.target_entropy).detach()).mean()
            else:
                ent_coef_loss = -(self.log_ent_coef * (action_log_probs + self.target_entropy).detach()).mean()
            self.ent_coef_optimizer.zero_grad(set_to_none=False)
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()
            ent_coef_dict["ent_coef_loss"] = ent_coef_loss.item()
            if self.ent_coef_scheduler is not None:
                self.ent_coef_scheduler.step()
        else:
            ent_coef = self.ent_coef_tensor
        ent_coef_dict["ent_coef"] = ent_coef.item()
        return ent_coef, ent_coef_dict

    def learn(
        self,
        total_timesteps: int,
        callback=None,
        log_interval: int = 4,
        eval_env=None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "DecisionTransformer",
        eval_log_path=None,
        reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:
        """
        Original learn() function with additional option to inititalize replay buffer with top k% of trajectories
        after initial interaction phase (i.e., < learning_starts).

        """
        buffer_reinit = True
        continue_training = self.load_kwargs.get("load_optim", False) if self.load_kwargs is not None else False
        if continue_training: 
            # make sure that saving/evaluation happens in right intervals
            for cb in callback.callbacks: 
                cb.n_calls = self.num_timesteps
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            eval_log_path,
            reset_num_timesteps if not continue_training else False,
            tb_log_name,
        )
        state_mean, state_std = self.replay_buffer.get_state_mean_std()
        if state_mean is not None and state_std is not None:
            self.state_mean = torch.from_numpy(state_mean).to(self.device).float()
            self.state_std = torch.from_numpy(state_std).to(self.device).float()
        
        self.policy.eval()
        callback.on_training_start(locals(), globals())
        self.policy.train()
        self._record_param_count()
        self._dump_logs()

        if self.offline_steps > 0:
            if len(self.replay_buffer) == 0:
                print("Populating replay buffer.")
                rollout = self.collect_rollouts(
                    self.env,
                    train_freq=TrainFreq(100, TrainFrequencyUnit.EPISODE),
                    action_noise=self.action_noise,
                    callback=callback,
                    learning_starts=self.learning_starts,
                    replay_buffer=self.replay_buffer,
                    log_interval=log_interval,
                )
            self.train(batch_size=self.batch_size, gradient_steps=self.gradient_steps)
            self.num_timesteps += 1
            self.offline_steps -= 1
            self.policy.eval()
            callback.on_step()
            self.policy.train()

        while self.num_timesteps < total_timesteps:
            if self.offline_steps > 0:
                rollout = RolloutReturn(0, 0, True)
                if log_interval is not None and self.num_timesteps % log_interval == 0:
                    self._dump_logs()
                if self.steps_per_task is not None and self.num_timesteps % self.steps_per_task == 0:
                    self._on_task_switch()
            else:
                rollout = self.collect_rollouts(
                    self.env,
                    train_freq=self.train_freq,
                    action_noise=self.action_noise,
                    callback=callback,
                    learning_starts=self.learning_starts,
                    replay_buffer=self.replay_buffer,
                    log_interval=log_interval,
                )

            if rollout.continue_training is False:
                break
            if self.num_timesteps > self.learning_starts and buffer_reinit and self.buffer_reinit_percent is not None:
                buffer_reinit = False
                self.replay_buffer.reset(self.buffer_reinit_percent)
            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts and len(self.replay_buffer):
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                if self.max_grad_steps is not None:
                    gradient_steps = min(gradient_steps, self.max_grad_steps)
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)
                if self.offline_steps > 0:
                    self.num_timesteps += 1
                    self.offline_steps -= 1
                    self.policy.eval()
                    callback.on_step()
                    self.policy.train()
                    if self.set_frozen_eval:
                        self.policy.apply(set_frozen_to_eval)

            if self.reinit_policy_after is not None and self.num_timesteps >= self.reinit_policy_after:
                print("Reinitializing policy.")
                self.policy.post_init()
                self._setup_policy()
                self._setup_critic()
                self._setup_entropy_tuning()
                self.reinit_policy_after = None

        callback.on_training_end()
        self._dump_logs()

        return self

    def _dump_logs(self) -> None:
        buffer_stats = self.replay_buffer._get_buffer_stats()
        for k, v in buffer_stats.items():
            self.logger.record(k, round(v, 2))
        for env in self.env.envs:
            # only used for CW20, no need to handle multiple envs.
            if hasattr(env, "pop_successes"):
                successes = env.pop_successes()
                if len(successes) > 1:
                    avg_success = np.mean(successes)
                    self.logger.record("rollout/success", avg_success)
        samples_seen = self._n_updates * self.batch_size
        if self.ddp:                 
            self.logger.record("global_rank", self.global_rank)
            samples_seen *= int(os.environ["WORLD_SIZE"])
        # samples per second + seen samples
        samples_seen_at_start = self._num_timesteps_at_start * self.batch_size
        sps = int((samples_seen - samples_seen_at_start) / ((time.time() - self.start_time) + 1e-8))
        self.logger.record("time/sps", sps)
        self.logger.record("train/samples_seen", samples_seen)
        if self.ddp: 
            self.logger.record("time/sps_per_device", int(sps / int(os.environ["WORLD_SIZE"])))
        super()._dump_logs()

    def _record_metrics_from_dict(self, metrics, prefix, aggregation=np.mean):
        for k, v in metrics.items():
            if prefix == "train" and (self.accumulation_steps > 1 or
                                      isinstance(self.replay_buffer, MultiDomainTrajectoryReplayBuffer)): 
                # make sure that metrices are averaged over the course of training
                # e.g., useful in multi-domain to get averaged loss instead of single loss points
                self.logger.record_mean(f"{prefix}/{k}", aggregation(v))
            else: 
                self.logger.record(f"{prefix}/{k}", aggregation(v))

    def _on_step(self):
        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        if self.exploration_sched != "constant":
            self.logger.record("rollout/exploration_rate", self.exploration_rate)
        if self.use_prompt_buffer and self.num_timesteps % self.prompt_buffer_sync_freq == 0:
            self._sync_prompt_buffer()
        if self.steps_per_task is not None and self.num_timesteps % self.steps_per_task == 0:
            self._on_task_switch()
            return False

    def _update_info_buffer(self, infos, dones=None) -> None:
        super()._update_info_buffer(infos, dones)
        for idx, info in enumerate(infos):
            # continual_world adds "success", but not is_success
            maybe_is_success = info.get("success")
            if maybe_is_success is not None and dones[idx]:
                self.ep_success_buffer.append(maybe_is_success)

    def _sync_prompt_buffer(self):
        trjs = self.replay_buffer._get_topk_trajectories(self.prompt_buffer.num_trjs_per_task)
        self.prompt_buffer.add_trajectory(trjs, task_id=self.current_task_id)

    def _update_eval_prompt(self):
        if not self.prompt_buffer.is_empty(self.current_task_id):
            # in case norm_obs is true, the prompt will get normalized. However, the normlaization params
            # are becoming old the more steps we take.
            self.eval_prompt = self.prompt_buffer.sample(batch_size=self.env.num_envs,
                                                         env=self._vec_normalize_env,
                                                         task_id=self.current_task_id)
            self.eval_prompt = self.eval_prompt._replace(rewards_to_go=self.eval_prompt.rewards_to_go / self.reward_scale,
                                                         rewards=self.eval_prompt.rewards / self.reward_scale)

    def _on_task_switch(self):
        if isinstance(self.replay_buffer, ContinualTrajectoryReplayBuffer):
            # offline learning
            task_id = self.num_timesteps // self.steps_per_task
            self.replay_buffer.set_task_id(task_id)
            self.current_task_id = task_id
        else:
            # Reinitialize buffer, after every task --> Continual RL
            self.replay_buffer.reset(0)
            self.current_task_id = self._extract_current_task_id(self.env)
        print(f"Switching task id to {self.current_task_id} at step {self.num_timesteps}")
        self.current_task_id_tensor = torch.tensor(self.current_task_id, device=self.device, dtype=torch.int32)
        self.ep_info_buffer = collections.deque(maxlen=200)
        self.ep_success_buffer = collections.deque(maxlen=200)
        self.eval_prompt = None
        if hasattr(self.policy, "prompt"):
            # self.policy.prompt.reset_counts(self.device)
            self.policy.prompt.set_task_id(self.current_task_id)
        if self.reset_optim_on_switch:
            print("Resetting optimizer.")
            params = self.policy.get_optim_groups(weight_decay=self.weight_decay) if self.weight_decay > 0 \
                else self.policy.parameters()
            self.optimizer = make_optimizer(self.optimizer_kind, params, lr=self.learning_rate)

    def _extract_current_task_id(self, env):
        temp_env = env.envs[0]
        if hasattr(temp_env, "cur_seq_idx"):
            return temp_env.cur_seq_idx
        return 0

    def _excluded_save_params(self):
        # "state_mean", "state_std"
        return super()._excluded_save_params() + ["policy_class", "lr_schedule",
                                                  "schedulers", "loss_fn", "exploration_scheduler",
                                                  "log_ent_coef", "ent_coef_optimizer", "ent_coef_scheduler",
                                                  "prompt_buffer", "prompt.counts", "prompt.inv_counts_so_far"]

    def _get_torch_save_params(self):
        torch_vars = ["state_mean", "state_std"]
        state_dicts = ["policy", "optimizer"]
        return state_dicts, torch_vars

    def load_model_weights(self, path, freeze=False):
        print(f"Loading model weights from: {path}")
        exclude_params = ["prompt.counts", "prompt.inv_counts_so_far"]
        load_kwargs = self.load_kwargs if self.load_kwargs is not None else {}
        if not load_kwargs.get("load_action_head", True):
            base_names = ["action_net", "action_pred", "mu", "log_std"]
            for name in base_names:
                exclude_params += [f"{name}.weight", f"{name}.bias"]
                exclude_params += [f"{name}.0.weight", f"{name}.0.bias"]
                exclude_params += [f"{name}.1.weight", f"{name}.1.bias"]
        if not load_kwargs.get("load_prompt", True):
            exclude_params.append("prompt.prompt")
        if not load_kwargs.get("load_prompt_keys", True):
            exclude_params.append("prompt.pretrain_key")
        if not load_kwargs.get("load_state_head", False):
            exclude_params += ["predict_state.weight", "predict_state.bias"]

        # load params
        _, params, variables = load_from_zip_file(path, device=self.device)
        policy_dict = params["policy"]
        # models may be saved with "module." prefix, replace 
        policy_dict = {k.replace("module.", "", 1): v for k, v in policy_dict.items() 
                       if k.replace("module.", "", 1) not in exclude_params}
        missing_keys, unexpected_keys = self.policy.load_state_dict(policy_dict, strict=False)
        if missing_keys:
            print("Missing key(s) in state_dict:", missing_keys)
        if unexpected_keys:
            print("Unexpected key(s) in state_dict:", unexpected_keys)
        if self.policy.num_task_heads > 1:
            # copy original mu params to task heads
            print("Copying weights original action head weights to new action heads.")
            self.policy.load_action_head_weights(policy_dict)
        if freeze:
            freeze_kwargs = self.freeze_kwargs if self.freeze_kwargs is not None else {}
            frozen, not_frozen = self.policy.freeze(**freeze_kwargs)
            print("Frozen layers:", frozen)
            print("Trainable layers:", not_frozen)
        if "state_mean" in variables and "state_std" in variables:
            self.state_mean = variables["state_mean"]
            self.state_std = variables["state_mean"]

    def load_optim_state(self, path):
        print(f"Loading optimizer state: {path}")
        data, params, _ = load_from_zip_file(path, device=self.device)
        self.optimizer.load_state_dict(params["optimizer"])
        print(f"Continuing training at step: {data['num_timesteps']}")
        self._n_updates = data["_n_updates"]
        self.num_timesteps = data["num_timesteps"]

    def _record_param_count(self):
        counts = get_param_count(self.policy, "policy")
        if self.policy.num_task_heads > 1:
            if self.stochastic_policy:
                mu_counts = get_param_count(self.policy.mu, "task_head_mu")
                logstd_counts = get_param_count(self.policy.log_std, "task_head_log_std")
                counts = {**counts, **mu_counts, **logstd_counts}
            else:
                head_counts = get_param_count(self.policy.action_pred, "task_head_action_pred")
                counts = {**counts, **head_counts}
        total = sum([v for k, v in counts.items() if "total" in k])
        trainable = sum([v for k, v in counts.items() if "trainable" in k])
        counts["percent_trainable"] = trainable / total * 100
        if hasattr(self.policy, "prompt"):
            prompt_counts = get_param_count(self.policy.prompt, "prompt")
            prompt_total = prompt_counts["prompt_total"]
            percent_prompt = prompt_total / total * 100
            percent_wo_prompt = prompt_total / (total - prompt_total) * 100
            counts = {**counts, **prompt_counts, "percent_prompt": percent_prompt, 
                      "percent_wo_prompt": percent_wo_prompt}
        for k, v in counts.items():
            self.logger.record(f"param_counts/{k}", v)

    def _record_prompt_infos(self, prompt_infos, prefix="prompt"):
        prompt_dict = {}
        if "reduce_sim" in prompt_infos:
            prompt_dict[f"similarity"] = prompt_infos['reduce_sim'].item()
            prompt_dict[f"total_prompt_len"] = prompt_infos['total_prompt_len']
        if "prompt_key" in prompt_infos:
            prompt_dict[f"prompt_key_mean"] = prompt_infos['prompt_key'].mean().item()
            prompt_dict[f"prompt_key_std"] = prompt_infos['prompt_key'].std().item()
            prompt_dict[f"selected_key_mean"] = prompt_infos['selected_key'].mean().item()
            prompt_dict[f"selected_key_std"] = prompt_infos['selected_key'].std().item()
            prompt_dict[f"x_embed_mean"] = prompt_infos['x_embed_mean'].mean().item()
            prompt_dict[f"x_embed_std"] = prompt_infos['x_embed_mean'].std().item()
        if "mod_k_mean_0" in prompt_infos:
            for k in prompt_infos.keys():
                if k.startswith("mod_"):
                    prompt_dict[k] = prompt_infos[k]
        for k, v in prompt_dict.items():
            self.logger.record(f"{prefix}/{k}", v)
        # compute counts for the selected prompts.
        if "prompt_idx" in prompt_infos and self._n_updates % 10000 == 0 and not self.debug:
            if hasattr(self.policy.prompt, "counts"):
                counts = self.policy.prompt.counts.cpu().numpy()
                fig1, fig2 = make_promptcount_figures(counts, self.num_timesteps)
                self.logger.record(f"{prefix}/select_ratio", Figure(fig2, True), exclude="stdout")
        if self._n_updates % 20000 == 0 and not self.debug:
            if "similarity" in prompt_infos:
                similarity = prompt_infos["similarity"].detach().cpu().numpy()
                fig = make_sim_plot(similarity, self.num_timesteps)
                self.logger.record(f"{prefix}/similarity", Figure(fig, True), exclude="stdout")
        return prompt_dict

    def _record_attention_maps(self, attention_scores, prefix="train", step=None,
                               lower_triu=True, vmin=None, vmax=None):
        if step is None:
            step = self.num_timesteps
        figures = make_attention_maps(attention_scores, step, lower_triu=lower_triu, vmin=vmin, vmax=vmax)
        for k, v in figures.items():
            self.logger.record(f"{prefix}_attn_scores/{k}", Figure(v, True), exclude="stdout")
