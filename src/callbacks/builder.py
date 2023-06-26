import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, StopTrainingOnNoModelImprovement
from ..algos import AGENT_CLASSES
from ..envs.env_names import ID_TO_NAMES, MT50_ENVS, MT50_ENVS_v2, ATARI_ENVS, DM_CONTROL_ENVS
from .custom_eval_callback import CustomEvalCallback, MultiEnvEvalCallback


class CustomWandbCallback(WandbCallback):

    def __init__(self, model_sync_wandb=False, **kwargs):
        super().__init__(**kwargs)
        self.model_sync_wandb = model_sync_wandb

    def save_model(self) -> None:
        self.model.save(self.path)
        if self.model_sync_wandb:
            wandb.save(self.path, base_path=self.model_save_path)
            print(f"Saving model checkpoint to {self.path}")


def make_callbacks(config, eval_env=None, logdir=None):
    callbacks = []
    if config.use_wandb and logdir is not None:
        model_save_path = None
        if config.wandb_callback_params.model_save_path is not None:
            model_save_path = f"{logdir}/{config.wandb_callback_params.model_save_path}"
        callbacks.append(
            CustomWandbCallback(
                gradient_save_freq=config.wandb_callback_params.gradient_save_freq,
                verbose=config.wandb_callback_params.verbose, model_save_path=model_save_path,
                model_sync_wandb=config.wandb_callback_params.get("model_sync_wandb", False),
                model_save_freq=config.wandb_callback_params.get("model_save_freq", 0)
            )
        )
    if config.eval_params.use_eval_callback:
        if config.agent_params.kind in AGENT_CLASSES.keys():
            if config.env_params.envid not in [*list(ID_TO_NAMES.keys()), *ATARI_ENVS, 
                                               *MT50_ENVS, *MT50_ENVS_v2, *DM_CONTROL_ENVS]:
                eval_callback_class = CustomEvalCallback
            else:
                eval_callback_class = MultiEnvEvalCallback
        else:
            eval_callback_class = EvalCallback
        if config.eval_params.max_no_improvement_evals > 0:
            stop_training_callback = StopTrainingOnNoModelImprovement(
                max_no_improvement_evals=config.eval_params.max_no_improvement_evals, verbose=1)
        else:
            stop_training_callback = None
        eval_callback = eval_callback_class(
            eval_env=eval_env, n_eval_episodes=config.eval_params.n_eval_episodes,
            eval_freq=config.eval_params.eval_freq, callback_after_eval=stop_training_callback,
            deterministic=config.eval_params.deterministic, first_step=config.eval_params.get("first_step", True)
        )
        callbacks.append(eval_callback)
    return CallbackList(callbacks)
