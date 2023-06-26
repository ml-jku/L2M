"""
Adapted from: https://github.com/adapter-hub/adapter-transformers/blob/master/src/transformers/adapters/modeling.py

"""
import math
import torch
from torch import nn
from transformers.activations import get_activation


class Activation_Function_Class(nn.Module):
    """
    Implementation of various activation function.
    """

    def __init__(self, hidden_act):
        super().__init__()
        if hidden_act.lower() == "leakyrelu":
            self.f = nn.functional.leaky_relu
        else:
            self.f = get_activation(hidden_act.lower())

    def forward(self, x):
        return self.f(x)


class Adapter(nn.Module):
    """
    Implementation of a sequential bottleneck adapter block.
    Parameters from: https://github.com/adapter-hub/adapter-transformers/blob/4530d8cf1aff7403d41919c9a7f5ceacd0ca7f60/src/transformers/adapters/configuration.py#L301

    """

    def __init__(
        self,
        input_size,
        down_sample=None,
        original_ln_before: bool = False,
        original_ln_after: bool = True,
        residual_before_ln: bool = True,
        adapter_residual_before_ln: bool = False,
        ln_before: bool = False,
        ln_after: bool = False,
        mh_adapter: bool = True,
        output_adapter: bool = True,
        use_gating: bool = False,
        non_linearity: str = "swish",
        init_weights: str = "bert",
        reduction_factor: float = 16,
        scaling: float = 1.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.down_sample = down_sample
        self.original_ln_before = original_ln_before
        self.original_ln_after = original_ln_after
        self.residual_before_ln = residual_before_ln
        self.adapter_residual_before_ln = adapter_residual_before_ln
        self.ln_before = ln_before
        self.ln_after = ln_after
        self.mh_adapter = mh_adapter
        self.output_adapter = output_adapter
        self.reduction_factor = reduction_factor
        self.scaling = scaling
        self.init_weights = init_weights
        self.use_gating = use_gating

        # list for all modules of the adapter, passed into nn.Sequential()
        seq_list = []

        # If we want to have a layer norm on input, we add it to seq_list
        if self.ln_before:
            self.adapter_norm_before = nn.LayerNorm(self.input_size)
            seq_list.append(self.adapter_norm_before)

        # if a downsample size is not passed, we just half the size of the original input
        if self.reduction_factor is not None:
            self.down_sample = self.input_size // self.reduction_factor
        else:
            self.down_sample = self.input_size // 2

        # ensure that the down sample size is at least 1
        if self.down_sample < 1:
            self.down_sample = 1

        seq_list.append(nn.Linear(self.input_size, self.down_sample))

        # select non-linearity
        self.non_linearity = Activation_Function_Class(non_linearity.lower())

        seq_list.append(self.non_linearity)

        # sequential adapter, first downproject, then non-linearity then upsample. In the forward pass we include the
        # residual connection
        self.adapter_down = nn.Sequential(*seq_list)

        # Up projection to input size
        self.adapter_up = nn.Linear(self.down_sample, self.input_size)

        # Additional scaling factor (from He et al. (2021))
        if isinstance(self.scaling, float):
            self.scaling = self.scaling
        elif self.scaling == "learned":
            self.scaling = nn.Parameter(torch.ones(1))
        else:
            raise ValueError("Unknown scaling type: {}".format(self.scaling))

        # If we want to have a layer norm on output, we apply it later after a separate residual connection
        # This means that we learn a new output layer norm, which replaces another layer norm learned in the bert layer
        if self.ln_after:
            self.adapter_norm_after = nn.LayerNorm(self.input_size)

        if self.use_gating:
            self.gate = nn.Linear(self.input_size, 1)

        # if we want to initialize with the bert strategy then this function is called for all the linear layers
        if self.init_weights == "bert":
            self.adapter_down.apply(self.init_bert_weights)
            self.adapter_up.apply(self.init_bert_weights)
            if self.use_gating:
                self.gate.apply(self.init_bert_weights)
        elif self.init_weights == "mam_adapter":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.adapter_down[0].weight, a=math.sqrt(5))
                nn.init.zeros_(self.adapter_up.weight)
                nn.init.zeros_(self.adapter_down[0].bias)
                nn.init.zeros_(self.adapter_up.bias)
                if self.use_gating:
                    self.gate.apply(self.init_bert_weights)
        else:
            raise ValueError("Unknown init_weights type: {}".format(self.init_weights))

    def pre_forward(
        self,
        hidden_states,
        input_tensor=None,
        layer_norm=None,
        fusion_config=None,
    ):
        """
        Retrieves the hidden_states, query (for Fusion), and residual connection according to the set configuration.
        Args:
            adapter_config: config file according to what the parameters are passed
            hidden_states: output of previous layer
            input_tensor: residual connection before FFN
        Returns: hidden_states, query, residual
        """
        query = None

        if self.residual_before_ln:
            residual = hidden_states

        if fusion_config is not None and fusion_config["query_before_ln"]:
            query = hidden_states

        if self.original_ln_before:
            if layer_norm:
                hidden_states = layer_norm(hidden_states + input_tensor)
            else:
                hidden_states = hidden_states + input_tensor

        if not self.residual_before_ln:
            residual = hidden_states

        if fusion_config is not None and not fusion_config["query_before_ln"]:
            query = hidden_states

        return hidden_states, query, residual

    def forward(self, x, residual_input=None, output_gating=False):
        down = self.adapter_down(x)
        up = self.adapter_up(down)
        up = up * self.scaling
        output = up

        if self.use_gating:
            # x.shape = (batch_size, seq_len, hidden_size)
            gate = torch.sigmoid(self.gate(x))
            gate = torch.mean(gate, dim=1).unsqueeze(-1)
            output = output * gate

        # apply residual connection before layer norm if configured in this way
        if self.adapter_residual_before_ln:
            output = output + residual_input

        # apply layer norm if available
        if self.ln_after:
            output = self.adapter_norm_after(output)

        # if residual should be applied after layer norm, apply it here
        if not self.adapter_residual_before_ln:
            output = output + residual_input

        if self.use_gating and output_gating:
            return output, down, up, gate
        return output, down, up

    def post_forward(self, hidden_states, input_hidden_states, input_tensor, layer_norm):
        """
        Performs computations after the forward pass of the adapter block(s). This e.g. includes applying the residual
        connection and layer norm if configured in this way.
        Args:
            hidden_states: The hidden states outputted by the adapter block(s).
            input_hidden_states: Residual connection before the adapter block(s).
            input_tensor: Residual connection before the Transformer FFN/ attention layer.
            layer_norm: Transformer LayerNorm.
        Returns:
            The modified hidden states.
        """
        if self.original_ln_after:
            if layer_norm:
                hidden_states = layer_norm(hidden_states + input_tensor)
            else:
                hidden_states = hidden_states + input_tensor

        return hidden_states

    # This is copied from the BertPreTrainedModel class to make this a self containing class.
    @staticmethod
    def init_bert_weights(module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # std defaults to 0.02, this might need to be changed
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
