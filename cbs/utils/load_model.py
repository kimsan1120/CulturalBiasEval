import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
from utils.file_utils import read_yaml
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[1] / 'configs'
MODEL_CONFIGS = read_yaml(CONFIG_DIR / 'model_configs.yaml')
USER_CONFIGS = read_yaml(CONFIG_DIR / 'user_configs.yaml')

def load_matching_model(model_name, use_deepspeed):
    model_config = MODEL_CONFIGS.get(model_name, None)
    assert model_config is not None, f"Model {model_name} not found in model_configs.yaml"

    model_full_name = model_config['full_name']
    model_type = model_config['model_type']

    # Configurations
    model_torch_dtype = model_config.get('torch_dtype', None)
    if model_torch_dtype == 'float16':
        model_torch_dtype = torch.float16
    elif model_torch_dtype == 'bfloat16':
        model_torch_dtype = torch.bfloat16
    elif model_torch_dtype == 'float32':
        model_torch_dtype = torch.float32
    else:
        raise ValueError(f"Invalid torch_dtype {model_torch_dtype}")
    
    # Load model
    hf_config = USER_CONFIGS.get('hf_config', None)
    if model_type == 'decoder':
        tokenizer = AutoTokenizer.from_pretrained(
            model_full_name,
            token = hf_config.get('access_token', None),
            padding_side = 'left',
            )

        tokenizer.pad_token = tokenizer.eos_token
        # DeepSpeed
        if use_deepspeed:
            model = AutoModelForCausalLM.from_pretrained(
                model_full_name,
                torch_dtype = model_torch_dtype,
                device_map = model_config.get('device_map', None),
                token = hf_config.get('access_token', None)
                )
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = AutoModelForCausalLM.from_pretrained(
                model_full_name,
                torch_dtype = model_torch_dtype,
                device_map = device,
                token = hf_config.get('access_token', None),
                trust_remote_code=True
                )
    elif model_type == 'encoder':
        tokenizer = AutoTokenizer.from_pretrained(model_full_name)
        # DeepSpeed
        if use_deepspeed:
            model = AutoModelForMaskedLM.from_pretrained(
                model_full_name,
                torch_dtype = model_torch_dtype,
                device_map = model_config.get('device_map', None)
            )
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = AutoModelForMaskedLM.from_pretrained(
                model_full_name,
                torch_dtype = model_torch_dtype,
                device_map = device
            )
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")
    
    return model, tokenizer, model_config