import torch
import torch.nn as nn
import numpy as np
from transformers import GPT2Model
from .online_decision_transformer_model import OnlineDecisionTransformerModel


class HelmDTModel(OnlineDecisionTransformerModel):

    def __init__(self, config, observation_space, action_space, beta=1,
                 on_actions=True, on_states=False, on_returns=True,
                 pretrained_lm="distilgpt2", train_hopfield=False, **kwargs):
        super().__init__(config, observation_space, action_space, **kwargs)
        self.on_actions = on_actions
        self.on_states = on_states
        self.on_returns = on_returns
        self.train_hopfield = train_hopfield
        self.encoder.from_pretrained(pretrained_lm, config=config, ignore_mismatched_sizes=True)
        self.embedding_model = GPT2Model.from_pretrained(pretrained_lm)
        for p in self.embedding_model.parameters():
            p.requires_grad_(False)
        n_tokens = self.embedding_model.wte.num_embeddings
        embedding_dim = self.embedding_model.wte.embedding_dim
        embeddings = self.embedding_model.wte(torch.arange(n_tokens))
        del self.embedding_model
        if self.on_states:
            del self.embed_state
            self.embed_state = FrozenHopfield(embedding_dim, self.config.state_dim, embeddings,
                                              beta=beta, trainable=train_hopfield)
        if self.on_actions:
            del self.embed_action
            self.embed_action = FrozenHopfield(embedding_dim, self.config.act_dim, embeddings,
                                               beta=beta, trainable=train_hopfield)
        if self.on_returns:
            del self.embed_return
            self.embed_return = FrozenHopfield(embedding_dim, 1, embeddings,
                                               beta=beta, trainable=train_hopfield)

    def embed_inputs(self, states, actions, returns_to_go, rewards, attention_mask):
        batch_size, context_len = states.shape[0], states.shape[1]
        if self.on_states:
            # shape: batch_size x context_len x state_dim
            states = states.reshape(-1, self.config.state_dim)
            state_embeddings = self.embed_state(states)
            # reshape back: batch_size x context_len x embedding_dim
            state_embeddings = state_embeddings.reshape(batch_size, context_len, -1)
        else:
            state_embeddings = self.embed_state(states)
        if self.on_actions:
            # shape: batch_size x context_len x act_dim
            actions = actions.reshape(-1, self.config.act_dim)
            action_embeddings = self.get_action_embeddings(actions, attention_mask=attention_mask)
            # reshape back: batch_size x context_len x embedding_dim
            action_embeddings = action_embeddings.reshape(batch_size, context_len, -1)
        else:
            action_embeddings = self.get_action_embeddings(actions, attention_mask=attention_mask)

        if self.on_returns:
            # shape: batch_size x context_len x 1
            returns_to_go = returns_to_go.reshape(-1, 1)
            returns_embeddings = self.embed_return(returns_to_go)
            returns_embeddings = returns_embeddings.reshape(batch_size, context_len, -1)
        else:
            returns_embeddings = self.embed_return(returns_to_go)
        rewards = self.get_reward_embeddings(rewards)
        return state_embeddings, action_embeddings, returns_embeddings


class FrozenHopfield(nn.Module):
    def __init__(self, embedding_dim, input_dim, embeddings, beta, proj_dim=768, trainable=False):
        """
        Adjusted from: https://github.com/ml-jku/helm/blob/main/model.py

        """
        super(FrozenHopfield, self).__init__()
        self.beta = beta

        # # down project embeddings
        # self.embed_proj = torch.nn.Parameter(
        #     torch.normal(mean=0.0, std=1 / np.sqrt(proj_dim), size=(proj_dim, embedding_dim)),
        #     requires_grad=False
        # )
        # embeddings = embeddings @ self.embed_proj.T
        # https://discuss.pytorch.org/t/why-model-to-device-wouldnt-put-tensors-on-a-custom-layer-to-the-same-device/17964/11
        self.register_buffer("embeddings", embeddings)

        self.rand_proj = torch.nn.Parameter(
            torch.normal(mean=0.0, std=1 / np.sqrt(embedding_dim), size=(embedding_dim, input_dim)),
            requires_grad=trainable
        )

    def forward(self, x):
        x = x @ self.rand_proj.T
        similarities = x @ self.embeddings.T / (
                    x.norm(dim=-1).unsqueeze(1) @ self.embeddings.norm(dim=-1).unsqueeze(0) + 1e-8)
        softm = torch.softmax(self.beta * similarities, dim=-1)
        state = softm @ self.embeddings
        return state
