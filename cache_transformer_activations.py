from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import Dataset
from dataclasses import dataclass
from typing import List, Dict, Union, Tuple, Optional
import torch
from torch.utils.data import DataLoader
import tqdm

@dataclass
class ActivationCache:
    activations: Optional[Dict[int, List[torch.Tensor]]] = None

def cache_transformer_activations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    layers: List[int],
    batch_size: int = 128,
    truncate_length: Optional[int] = None,
    min_num_activations: int = 1_000_000,
    output_batch_size: int = 1024,
    p_sample: float = 0.1,
    device: str = "cuda",
) -> ActivationCache:
    model.to(device)
    model.eval()
    
    cache = ActivationCache(activations={layer: [] for layer in layers})

    total_activations = 0

    dataset.set_format(type="torch", columns=["input_ids"])
    batch_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    batch_iter = iter(batch_loader)
    processed_sentences = 0

    progress_bar = tqdm.tqdm(total=min_num_activations)
    while total_activations < min_num_activations:
        try:
            inputs = next(batch_iter)
            processed_sentences += len(inputs["input_ids"])
        except StopIteration:
            print("Ran out of data before reaching min_num_activations, returning early.")
            break
        # inputs = tokenizer(batch["text"], padding=True, truncation=True, max_length=truncate_length, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            act_mask = inputs["input_ids"] != tokenizer.pad_token_id
            batch_sample = torch.randperm(act_mask.sum().item())[:(act_mask.sum() * p_sample).int()]
            total_activations += batch_sample.size(0)
            outputs = model(**inputs, output_hidden_states=True)
            for layer in layers:
                activation = outputs.hidden_states[layer]
                activation = activation[act_mask][batch_sample]
                activation = activation.to(torch.float32)
                for i in range(activation.shape[0] // output_batch_size):
                    cache.activations[layer].append(activation[i * output_batch_size:(i + 1) * output_batch_size].detach().cpu())
            
            progress_bar.update(batch_sample.size(0))
    
    return cache, processed_sentences