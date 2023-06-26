from .decision_transformer_sb3 import DecisionTransformerSb3
from .universal_decision_transformer_sb3 import UDT
from .discrete_decision_transformer_sb3 import DiscreteDecisionTransformerSb3
from .continual_sac import ContinualSAC
from .decision_transformer_with_ewc_sb3 import UDTWithEWC, DiscreteUDTWithEWC
from .models import OnlineDecisionTransformerModel, DiscreteDTModel, HelmDTModel, \
    DummyUDTModel, CustomContinuousCritic, MultiHeadContinuousCritic, \
    MultiPromptDTModel, DTWithAdapter, MultiDomainDiscreteDTModel, MultiDomainDiscreteDTWithAdapter, \
    MDMPDTModel, DiscreteMPDTModel, DiscreteDTWithAdapter


MODEL_CLASSES = {
    "DT": OnlineDecisionTransformerModel,
    "ODT": OnlineDecisionTransformerModel,
    "UDT": OnlineDecisionTransformerModel,
    "DummyUDT": DummyUDTModel,
    "MPDT": MultiPromptDTModel,
    "DDT": DiscreteDTModel,
    "MDDT": MultiDomainDiscreteDTModel,
    "HelmDT": HelmDTModel,
    "DTWithAdapter": DTWithAdapter,
    "UDTWithEWC": OnlineDecisionTransformerModel,
    "MDUDTWithEWC": MultiDomainDiscreteDTModel,
    "MDDTWithAdapter": MultiDomainDiscreteDTWithAdapter,  
    "DDTWithAdapter": DiscreteDTWithAdapter,  
    "MDMPDT": MDMPDTModel,
    "DMPDT": DiscreteMPDTModel  

}

AGENT_CLASSES = {
    "DT": DecisionTransformerSb3,
    "ODT": DecisionTransformerSb3,
    "UDT": UDT,
    "DummyUDT": UDT,
    "MPDT": UDT,
    "DTWithAdapter": UDT,
    "DDT": DiscreteDecisionTransformerSb3,
    "MDDT": DiscreteDecisionTransformerSb3,
    "HelmDT": DecisionTransformerSb3,
    "UDTWithEWC": UDTWithEWC,
    "MDUDTWithEWC": DiscreteUDTWithEWC,
    "MDDTWithAdapter": DiscreteDecisionTransformerSb3,
    "DDTWithAdapter": DiscreteDecisionTransformerSb3,
    "MDMPDT": DiscreteDecisionTransformerSb3,
    "DMPDT": DiscreteDecisionTransformerSb3  
}


def get_model_class(kind):
    assert kind in MODEL_CLASSES, f"Unknown kind: {kind}"
    return MODEL_CLASSES[kind]


def get_agent_class(kind):
    assert kind in AGENT_CLASSES, f"Unknown kind: {kind}"
    return AGENT_CLASSES[kind]
