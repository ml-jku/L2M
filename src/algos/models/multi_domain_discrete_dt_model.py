import gym
import numpy as np
import torch
import torch.nn as nn
from .image_encoders import make_image_encoder
from .discrete_decision_transformer_model import DiscreteDTModel
from ...tokenizers_custom import make_tokenizer


class MultiDomainDiscreteDTModel(DiscreteDTModel):
    
    def __init__(self, config, observation_space, action_space, action_channels=256, discrete_actions=18,
                 state_dim=39, image_shape=(1,84,84), **kwargs):
        """
        Discrete DT version that supports multi-domain training.
        Different domains have different state and action spaces. This model takes care of that.
        Input observation space and action space arguments are irrelevant, as they only account for a single train env.
        Instead, the class demands an image shape and state dim, which are used to make persistent observation encoders. 
        All actions need to be discrete to be used with the discrete action embeddings.
        
        This class should only be used in offline setting, as online data collection is not currently supported.
        

        Args:
            config: Huggingface config.
            observation_space: gym.Space. 
            action_space: gym.Space
            image_shape: Tuple or List. Shape of image observations.
            state_dim: Int. Dimension/shape of state observations.
            discrete_actions: Int. Defaults to 18 (full Atari action space). Number of discrete actions. 
                Also used as shift for the action tokenizer.
                
        """
        self.discrete_actions = discrete_actions
        self.num_actions = discrete_actions + action_channels
        super().__init__(config, observation_space, action_space, action_channels=action_channels, **kwargs)
        
        # make persistent state/image encoders
        self.image_shape = image_shape
        self.state_dim = state_dim
        if self.image_shape is not None:
            # overwrite if exists
            if self.patch_size is not None: 
                self.setup_patch_encoder()
            else: 
                self.embed_image = make_image_encoder(
                    observation_space=gym.spaces.Box(0, 255, self.image_shape, dtype=np.uint8),                                
                    features_dim=config.hidden_size, encoder_kwargs=self.encoder_kwargs
                )
        if self.state_dim is not None and not self.tokenize_s: 
            del self.embed_state
            self.embed_state = torch.nn.Linear(self.state_dim, config.hidden_size)
            
        # make action tokenizer with shift
        assert self.tokenize_a or self.action_channels == 0, "If not tokenizing, action channels must be 0."
        if self.tokenize_a: 
            a_tok_kind = self.a_tok_kwargs.pop('kind', 'minmax')
            # add shift argument to shift tokenization to the right by num of discrete actions
            self.action_tokenizer = make_tokenizer(
                a_tok_kind, 
                {'vocab_size': self.action_channels, "shift": self.discrete_actions, **self.a_tok_kwargs}
            )
        
        # make universal action embeddings
        self.action_pad_token = self.num_actions if self.use_action_pad else None
        self.embed_action_disc = nn.Embedding(
            self.num_actions + 1, config.hidden_size, padding_idx=self.action_pad_token
        )
        # self.post_init()
        self.setup_prompt()

    def setup_policy(self):
        if self.stochastic_policy:
            raise NotImplementedError("Stochastic policy not implemented for multi-domain discrete DT.")
        if self.num_task_heads > 1:
            self.action_net = nn.ModuleList(
                [self.make_head(self.config.hidden_size, self.num_actions, self.n_layer_head)
                 for _ in range(self.num_task_heads)]
            )
        else:
            self.action_net = self.make_head(self.config.hidden_size,  self.num_actions, self.n_layer_head)

    def get_action_from_logits(self, action_logits): 
        if action_logits.shape[-2] == 1: 
            # safeguard for discrete action spaces to avoid selecting actions > num discrete actions
            # we assume discrete action spaces have action dim of 1
            action = torch.argmax(action_logits[..., :self.discrete_actions], dim=-1)
        else: 
            action = torch.argmax(action_logits, dim=-1)
        if self.tokenize_a and action.shape[-1] > 1: 
            action = self.inv_tokenize_actions(action)
        if len(action.shape) == 2:
            action = action.unsqueeze(0)
        return action
