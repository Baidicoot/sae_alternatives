from modelling_sae import BasicSAE, RICA, UntiedRICA, ShrinkageRICA, ConstrainedRICA, SoftplusSAE, NonNegativeRICA
from cache_transformer_activations import cache_transformer_activations, ActivationCache
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
import torch
from typing import List
import tqdm
import wandb
import os
import numpy as np
import dotenv

dotenv.load_dotenv()
hf_token = os.getenv("HUGGINGFACE_API_KEY")

def load_and_process_dataset(
    dataset_name: str,
    tokenizer: AutoTokenizer,
    max_length: int,
):
    """
    Load, tokenize, and pack the dataset into sequences of a fixed length.
    """

    dataset = load_dataset(dataset_name, split="train")
    
    # tokenize the entire dataset
    dataset = dataset.map(
        lambda x: tokenizer(x["text"]),
        batched=True,
        remove_columns=["text"],
    )

    def split_dataset(dataset, chunk_size):
        """
        Split the dataset into chunks of a fixed length.
        """
        chunks = []
        for i in tqdm.tqdm(range(len(dataset))):
            packed_entry = dataset[i]["input_ids"]
            for j in range(0, len(packed_entry), chunk_size):
                chunk = packed_entry[j:j + chunk_size]
                if len(chunk) == chunk_size:
                    chunks.append(chunk)
                else:
                    continue
        return chunks

    dataset_chunks = split_dataset(dataset, max_length)
    
    # reprocess into a hf dataset
    dataset = Dataset.from_dict({
        "input_ids": dataset_chunks,
    })
    dataset.set_format(type="torch", columns=["input_ids"])

    return dataset

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def move_everything_to_device(models, optimizers, device):
    for model in models:
        model.to(device)
    for optimizer in optimizers:
        optimizer_to(optimizer, device)

def train_on_cache(
    models: List[torch.nn.Module],
    optimizers: List[torch.optim.Optimizer],
    schedulers: List[torch.optim.lr_scheduler._LRScheduler],
    model_names: List[str],
    cache: ActivationCache,
    layer: int,
    device: str = "cuda",
    num_epochs: int = 1,
    batch_size: int = 4096,
    log_loss_interval: int = 10, # number of batches to log loss/average loss over
    start_batch_count: int = 0,
):
    assert layer in cache.activations, f"Layer {layer} not found in cache."

    for epoch in range(num_epochs):
        for model in models:
            model.train()
        
        cache_idx = 0
        batch_idx = 0
        progress_bar = tqdm.tqdm(total=len(cache.activations[layer]))

        sparsity_running_mean = [0 for _ in models]
        loss_running_mean = [0 for _ in models]
        mse_running_mean = [0 for _ in models]
        l1_running_mean = [0 for _ in models]

        cache_running_mean = torch.zeros_like(cache.activations[layer][0][0]).to(device)
        mse_scale_running_mean = 0

        while True:
            current_batch_size = 0
            current_batch = []
            while current_batch_size < batch_size:
                if cache_idx >= len(cache.activations[layer]):
                    break
                n_new_samples = cache.activations[layer][cache_idx].size(0)
                current_batch_size += n_new_samples
                current_batch.append(cache.activations[layer][cache_idx].to(device))
                cache_idx += 1
                progress_bar.update(1)

            if len(current_batch) == 0:
                break

            current_batch = torch.cat(current_batch, dim=0)[:batch_size].to(dtype=torch.float32)
            
            metrics = {"epoch": epoch}

            batch_mean = current_batch.mean(dim=0)

            cache_running_mean = (cache_running_mean * batch_idx + batch_mean) / (batch_idx + 1)
            mse_scale = torch.nn.functional.mse_loss(cache_running_mean.expand_as(current_batch), current_batch, reduction='mean').item()
            mse_scale_running_mean = (mse_scale_running_mean * batch_idx + mse_scale) / (batch_idx + 1)

            for model_idx, (model, optimizer, scheduler, name) in enumerate(zip(models, optimizers, schedulers, model_names)):
                def step():
                    optimizer.zero_grad()
                    loss, mse, l1, h, x_hat = model(current_batch)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    loss_running_mean[model_idx] += loss.item()
                    mse_running_mean[model_idx] += mse.item()
                    sparsity_running_mean[model_idx] += (h.abs() > 0).float().sum(dim=-1).mean().item()
                    l1_running_mean[model_idx] += l1.item()

                    metrics[name] = {}

                    if log_loss_interval != 0 and batch_idx % log_loss_interval == log_loss_interval - 1:
                        loss_running_mean[model_idx] /= log_loss_interval
                        metrics[name]["loss"] = loss_running_mean[model_idx]
                        mse_running_mean[model_idx] /= log_loss_interval
                        metrics[name]["mse"] = mse_running_mean[model_idx]
                        metrics[name]["scaled_mse"] = mse_running_mean[model_idx] / mse_scale_running_mean
                        sparsity_running_mean[model_idx] /= log_loss_interval
                        metrics[name]["sparsity"] = sparsity_running_mean[model_idx]
                        metrics[name]["lr"] = optimizer.param_groups[0]["lr"]
                        l1_running_mean[model_idx] /= log_loss_interval
                        metrics[name]["l1"] = l1_running_mean[model_idx]
                        loss_running_mean[model_idx] = 0
                        mse_running_mean[model_idx] = 0
                        sparsity_running_mean[model_idx] = 0
                
                step()

            if log_loss_interval != 0 and batch_idx % log_loss_interval == log_loss_interval - 1:
                wandb.log(metrics, step=batch_idx + start_batch_count, commit=True)

            batch_idx += 1
    
    return batch_idx

def interleave_training_and_generation(
    models: List[torch.nn.Module],
    optimizers: List[torch.optim.Optimizer],
    schedulers: List[torch.optim.lr_scheduler._LRScheduler],
    model_names: List[str],
    tokenizer: AutoTokenizer,
    model_kwargs: dict,
    dataset_name: str,
    device: str = "cuda",
    num_epochs: int = 1,
    num_cache_epochs: int = 1,
    sample_batch_size: int = 128,
    sample_max_length: int = 512,
    train_batch_size: int = 4096,
    log_interval: int = 10,
    layer: int = 8,
):
    wandb.init(project="sae_alternatives", entity="baidicoot")

    tokenizer.pad_token_id = 0

    total_steps = 0

    dataset = load_and_process_dataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        max_length=sample_max_length,
    )

    print("Total # tokens in dataset:", len(dataset) * sample_max_length)

    total_processed_sentences = 0

    for epoch in range(num_epochs):
        move_everything_to_device(models, optimizers, "cpu")

        model = AutoModelForCausalLM.from_pretrained(**model_kwargs).to(device)

        cache, processed_sentences = cache_transformer_activations(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            layers=[layer],
            batch_size=sample_batch_size,
            output_batch_size=train_batch_size,
            truncate_length=sample_max_length,
            min_num_activations=10_000_000,
            p_sample=1,
            device=device,
        )

        del model

        move_everything_to_device(models, optimizers, device)

        total_steps += train_on_cache(
            models=models,
            optimizers=optimizers,
            schedulers=schedulers,
            model_names=model_names,
            cache=cache,
            layer=layer,
            device=device,
            num_epochs=num_cache_epochs,
            batch_size=train_batch_size,
            log_loss_interval=log_interval,
            start_batch_count=total_steps,
        )

        # save the models
        print("Saving.")
        for model, name in zip(models, model_names):
            os.makedirs("models", exist_ok=True)
            os.makedirs(f"models/epoch_{epoch}", exist_ok=True)
            torch.save(model.state_dict(), f"models/epoch_{epoch}/{name}.pt")
        
        total_processed_sentences += processed_sentences

        if total_processed_sentences >= len(dataset):
            print("Processed all sentences, restarting from the beginning.")
            total_processed_sentences = 0
    
    wandb.finish()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    l1_alphas = [1e-5, 1e-3, 1e-1]

    hidden_size = 2048
    ratio = 16

    models = [
        UntiedRICA(hidden_size, hidden_size * ratio, alpha) for alpha in l1_alphas
    ] + [
        RICA(hidden_size, hidden_size * ratio, alpha) for alpha in l1_alphas
    ]

    models = [model.to(device) for model in models]

    optimizers = [torch.optim.Adam(model.parameters(), lr=4e-4) for model in models]

    # linear warmup
    def linear_warmup(num_warmup_steps):
        def f(epoch):
            if epoch < num_warmup_steps:
                return epoch / num_warmup_steps
            return 1

        return f
    
    def linear_warmup_cooldown(num_warmup_steps, num_cooldown_steps):
        def f(epoch):
            if epoch < num_warmup_steps:
                return epoch / num_warmup_steps
            if epoch < num_warmup_steps + num_cooldown_steps:
                return max(1 - (epoch - num_warmup_steps) / num_cooldown_steps, 0)
            return 0

        return f

    schedulers = [
        torch.optim.lr_scheduler.LambdaLR(optimizer, linear_warmup_cooldown(1000, 100_000))
        for optimizer in optimizers
    ]

    model_names = [
        f"UntiedRICA_{alpha:.2e}" for alpha in l1_alphas
    ] + [
        f"RICA_{alpha:.2e}" for alpha in l1_alphas
    ]

    model_kwargs = {
        "pretrained_model_name_or_path": "EleutherAI/pythia-1.4b",
        "torch_dtype": torch.bfloat16,
        "token": hf_token,
    }

    interleave_training_and_generation(
        models=models,
        optimizers=optimizers,
        schedulers=schedulers,
        model_names=model_names,
        tokenizer=AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b"),
        model_kwargs=model_kwargs,
        dataset_name="Elriggs/openwebtext-100k",
        device=device,
        num_epochs=10,
        num_cache_epochs=1,
        sample_batch_size=32,
        sample_max_length=256,
        train_batch_size=4096,
        layer=16,
    )