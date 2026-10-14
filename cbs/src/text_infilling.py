import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1])) # /mnt/bias/ours
from tqdm import tqdm
from loguru import logger
import argparse

import torch
import torch.nn.functional as F
import deepspeed

from utils.load_model import load_matching_model
from utils.load_dataset import TextInfillingDataset

from utils.file_utils import append_jsonl, write_json

RESULT_DIR = Path(__file__).resolve().parents[1] / 'results' / 'text_infilling'

import torch
import torch.nn.functional as F
from tqdm import tqdm
from loguru import logger
from PIL import Image

def run_decoder(model, tokenizer, dataset, batch_size, args):
    all_probs = []

    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i+batch_size]
        batch_prompts = [datum['input']['combined_prompt'] for datum in batch]
        batch_entity_ids = [datum['input']['entity_ids'] for datum in batch]
        # TODO: Deepspeed check (local_rank)
        device = next(model.parameters()).device
        encoded_prompts = tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model(**encoded_prompts, output_hidden_states=True)
            logits = outputs.logits

        for idx, logit in enumerate(logits):
            entity_ids = batch_entity_ids[idx]
            entity_len = len(entity_ids)
            entity_logits = logit[-entity_len:]
            entity_probs = F.softmax(entity_logits, dim=-1)
            token_probs = entity_probs[torch.arange(entity_len), entity_ids]
            average_prob = torch.mean(token_probs).item()
            all_probs.append(average_prob)

    return all_probs

def run_encoder(model, tokenizer, dataset, batch_size, args):
    all_probs = []

    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i+batch_size]
        batch_prompts = [datum['input']['masked_prompt'] for datum in batch]
        entity_token_lists = [datum['input']['entity_tokens'] for datum in batch]

        device = next(model.parameters()).device
        encoded_prompts = tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=True).to(device)

        mask_positions_batch = []
        for input_ids in encoded_prompts['input_ids']:
            positions = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
            mask_positions_batch.append(positions)

        with torch.no_grad():
            outputs = model(**encoded_prompts)
            logits = outputs.logits
        
        for idx, mask_positions in enumerate(mask_positions_batch):
            entity_tokens = entity_token_lists[idx]
            entity_token_ids = tokenizer.convert_tokens_to_ids(entity_tokens)
            average_probs = []
            for j, pos in enumerate(mask_positions[:len(entity_token_ids)]):
                token_logits = logits[idx, pos, :]
                token_probs = F.softmax(token_logits, dim=-1)
                token_prob = token_probs[entity_token_ids[j]].item()
                average_probs.append(token_prob)
            all_probs.append(sum(average_probs) / len(average_probs))

    return all_probs


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='XLMR-Large', help='Model name in model_configs.yaml')
    parser.add_argument('--prompt_culture', type=str, default='co', help='Prompt culture in prompt dataset: (co, ag)')
    parser.add_argument('--language', type=str, default='ko', help='Language of the text', choices=['ko', 'en'])
    parser.add_argument('--prompt_sample_count', type=int, default=50, help='Number of prompts to sample')
    parser.add_argument('--entity_sample_count', type=int, default=50, help='Number of entities to sample')
    parser.add_argument('--special_name', type=str, default=None, help='Special name for the experiment')
    parser.add_argument('--entity_type', type=str, default=None, help='Entity type to run')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank passed by DeepSpeed")
    parser.add_argument("--use_deepspeed", action='store_true', help="Use DeepSpeed")
    parser.add_argument('--check_prev', action='store_true', help='Check previous results')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()

    # Result file setup
    experiment_name = f'{args.model_name}_{args.prompt_culture}_{args.language}_{args.prompt_sample_count}_{args.entity_sample_count}'
    if args.special_name is not None:
        experiment_name = f'{experiment_name}_{args.special_name}'
    if args.entity_type is not None:
        experiment_name = f'{experiment_name}_{args.entity_type}'
    experiment_file = RESULT_DIR / f'{experiment_name}.json'
    logger.info(f'Saving results to {experiment_file}')

    # Check prev results
    if args.check_prev and os.path.exists(experiment_file):
        logger.info(f'File {experiment_file} already exists. Skipping...')
        sys.exit(0)

    # Load model & use deepspeed
    model, tokenizer, model_config = load_matching_model(args.model_name, args.use_deepspeed)
    if args.use_deepspeed:
        model, _, _, _ = deepspeed.initialize(
            model = model, 
            model_parameters = model.parameters(),
            config = model_config['ds_config']
        )
    # Load dataset
    Dataset = TextInfillingDataset(args.prompt_culture, model_config['model_type'], args.language)

    # Run
    all_results = {}
    all_entity_types = Dataset.get_all_entity_types()
    for entity_type in all_entity_types:
        if args.entity_type is not None and entity_type != args.entity_type:
            continue
        dataset = Dataset.create_dataset(entity_type, tokenizer, args.entity_sample_count, args.prompt_sample_count)
        logger.debug(f'Running inference for {entity_type} with {len(dataset)} samples')
        if model_config['model_type'] == 'encoder':
            all_probs = run_encoder(model, tokenizer, dataset, args.batch_size, args)
        elif model_config['model_type'] == 'decoder':
            all_probs = run_decoder(model, tokenizer, dataset, args.batch_size, args)
        else:
            raise ValueError(f'Unknown model type: {model_config["model_type"]}')
        # Run inference
        logger.debug(f'Inference done for {entity_type} with {len(all_probs)} samples')
        all_results[entity_type] = {
            'dataset': dataset,
            'all_probs': all_probs
        }
    # Save
    write_json(all_results, experiment_file)
    
        