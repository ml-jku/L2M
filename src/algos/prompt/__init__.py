from .dummy_prompt import DummyPrompt
from .hopfield_prompt import HopfieldPrompt
from .ia3 import IA3
from .lora import LoRA
from .l2p import L2PPrompt
from .l2m import L2MIA3, L2MLoRA


def make_prompt(config, prompt_kwargs):
    prompt_kwargs = {} if prompt_kwargs is None else prompt_kwargs.copy()
    prompt_kind = prompt_kwargs.pop("kind", "l2p")
    if prompt_kind == "hopfield":
        prompt = HopfieldPrompt(embed_dim=config.hidden_size, pattern_size=config.hidden_size, **prompt_kwargs)
    elif prompt_kind == "dummy":
        prompt = DummyPrompt(embed_dim=config.hidden_size, **prompt_kwargs)
    elif prompt_kind == "ia3":
        prompt = IA3(embed_dim=config.hidden_size, **prompt_kwargs)
    elif prompt_kind == "lora":
        prompt = LoRA(embed_dim=config.hidden_size, **prompt_kwargs)
    elif prompt_kind == "l2m_ia3":
        prompt = L2MIA3(embed_dim=config.hidden_size, **prompt_kwargs)
    elif prompt_kind == "l2m_lora":
        prompt = L2MLoRA(embed_dim=config.hidden_size, **prompt_kwargs)
    else:
        prompt = L2PPrompt(embed_dim=config.hidden_size, **prompt_kwargs)
    return prompt
