from .vip_llava import VIPLLaVAMedCLM

def create_model(model_name: str, **kwargs):
    if 'llava' in model_name.lower() or 'vip' in model_name.lower():
        return VIPLLaVAMedCLM(model_name, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_name}")
