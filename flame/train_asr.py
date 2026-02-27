#!/usr/bin/env python3
"""
Unified ASR Training Framework for FLA Models with Built-in CTC Heads.
"""

import os
import gc
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from datetime import datetime
from pathlib import Path
import math

import torch
import torch.nn as nn
import torch.distributed as dist

from flame.config_manager import JobConfig, TORCH_DTYPE_MAP
from flame.models.parallelize_fla import parallelize_fla
from flame.logging_timeseries import TimeSeriesLogger
# from sparse_mamba.custom_dataloaders import create_librosa_raw_classification_dataset
from sparse_mamba.custom_dataloaders.librosa import create_librosa_raw_classification_dataset

# CHANGE 1 of 2: import the registry — this replaces the old build_hgrn_asr_model
# function and register_asr_model(ASRTrainSpec(...)) block entirely.
from flame.asr_registry import build_model

from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.components.ft import init_ft_manager, FTParallelDims
from torchtitan.tools.logging import logger as titan_logger
from torchtitan.tools import utils

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    
    
def Reg_Schedular(step, total_steps, start_value, target_value):
    """
    S-Curve Hyperparameter Scheduler
    Slow start -> Fast middle -> Slow end
    """
    if step >= total_steps:
        return target_value
    progress = step / total_steps
    factor = 0.5 * (1 - math.cos(math.pi * progress))
    return start_value + (target_value - start_value) * factor


# ============================================================================
# METRICS AND DECODING  (unchanged)
# ============================================================================

def levenshtein_distance(seq1, seq2):
    if len(seq1) < len(seq2):
        return levenshtein_distance(seq2, seq1)
    if not seq2:
        return len(seq1)
    prev = list(range(len(seq2) + 1))
    for i, c1 in enumerate(seq1):
        curr = [i + 1]
        for j, c2 in enumerate(seq2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def calculate_asr_metrics(predictions, targets, idx_to_char=None):
    metrics = {
        'cer': 0.0, 'wer': 0.0, 'exact_match': 0.0, 'token_accuracy': 0.0,
        'total_chars': 0, 'total_words': 0, 'total_sequences': len(predictions)
    }
    if not predictions:
        return metrics

    tce = twe = tc = tw = em = tk_c = tk_t = 0
    for pred, target in zip(predictions, targets):
        pred   = pred.tolist() if torch.is_tensor(pred)   else list(pred)
        target = target.tolist() if torch.is_tensor(target) else list(target)

        # CER — filter blanks, compare character tokens
        pc  = [p for p in pred   if p != 0]
        tc_ = [t for t in target if t != 0]
        tce += levenshtein_distance(pc, tc_)
        tc  += len(tc_)

        # WER — convert to text, split into words, compare word sequences
        if idx_to_char:
            #  pred_text   = "".join([idx_to_char.get(t, "") for t in pc]).replace("▁", " ").strip()
            pred_text   = "".join([idx_to_char.get(t, "") for t in pc]).replace("▁", " ").strip()
            target_text = "".join([idx_to_char.get(t, "") for t in tc_]).replace("▁", " ").strip()
            pred_words   = pred_text.split()
            target_words = target_text.split()
        else:
            # fallback if no idx_to_char — WER will equal CER
            pred_words   = pc
            target_words = tc_

        twe += levenshtein_distance(pred_words, target_words)
        tw  += len(target_words)

        # exact match and token accuracy unchanged
        em   += pc == tc_
        ml    = min(len(pc), len(tc_))
        tk_c += sum(pc[i] == tc_[i] for i in range(ml))
        tk_t += max(len(pc), len(tc_))

    metrics.update(
        cer=tce / max(tc, 1),
        wer=twe / max(tw, 1),
        exact_match=em / len(predictions),
        token_accuracy=tk_c / max(tk_t, 1),
        total_chars=tc,
        total_words=tw,
    )
    return metrics



def decode_predictions_ctc(logits, input_lengths, blank_id=0):
    preds = torch.argmax(logits, dim=-1) if logits.dim() == 3 else logits
    results = []
    for b in range(preds.size(0)):
        seq, decoded, prev = preds[b, :input_lengths[b].item()].tolist(), [], None
        for t in seq:
            if t != blank_id and t != prev:
                decoded.append(t)
            prev = t
        results.append(decoded)
    return results


# ============================================================================
# DATALOADER
# ============================================================================

def build_asr_dataloaders(job_config):
    dataset     = getattr(job_config.training, 'dataset',      'librispeech')
    max_samples = getattr(job_config.training, 'max_samples',  -1)
    num_mfcc    = getattr(job_config.training, 'num_mfcc',     80)
    cache_dir   = getattr(job_config.training, 'cache_dir',    './cache')
    spm_model_path = getattr(job_config.training, 'spm_model_path', None)

    train_loader, val_loader, test_loader, n_classes, seq_len, input_dim, char_to_idx, idx_to_char = \
        create_librosa_raw_classification_dataset(
            bsz=job_config.training.batch_size,
            max_samples=max_samples, num_mfcc=num_mfcc, cache_dir=cache_dir,
            dataset=dataset,spm_model_path=spm_model_path, drop_last=True, pin_memory=True,
            num_workers=job_config.training.num_workers,
        )

    for loader in (train_loader, val_loader, test_loader):
        loader.n_classes   = n_classes
        loader.seq_len     = seq_len
        loader.input_dim   = input_dim
        loader.char_to_idx = char_to_idx
        loader.idx_to_char = idx_to_char

    return train_loader, val_loader, test_loader


# ============================================================================
# OPTIMIZER / SCHEDULER  (unchanged)
# ============================================================================

def build_asr_optimizer(model_parts, job_config):
    params = [p for m in model_parts for p in m.parameters() if p.requires_grad]
    name = getattr(job_config.optimizer, 'name', 'AdamW')
    cls  = {'AdamW': torch.optim.AdamW, 'Adam': torch.optim.Adam}.get(name)
    if cls is None:
        raise ValueError(f"Unknown optimizer: {name}")
    return cls(params,
               lr=getattr(job_config.optimizer, 'lr', 3e-3),
               weight_decay=getattr(job_config.optimizer, 'weight_decay', 0.01),
               betas=(getattr(job_config.optimizer, 'beta1', 0.9),
                      getattr(job_config.optimizer, 'beta2', 0.95)),
               eps=getattr(job_config.optimizer, 'eps', 1e-8))


def build_asr_scheduler(optimizer, job_config):
    from transformers import get_cosine_schedule_with_warmup
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=job_config.lr_scheduler.warmup_steps,
        num_training_steps=job_config.training.steps,
    )


# ============================================================================
# TRAINING STATE  (unchanged)
# ============================================================================

@dataclass
class ASRTrainState:
    step: int = 0
    epoch: int = 0
    best_val_cer: float = float('inf')
    global_avg_losses: List[float] = field(default_factory=list)
    global_max_losses: List[float] = field(default_factory=list)
    log_steps: List[int] = field(default_factory=list)

    def state_dict(self):
        return {k: getattr(self, k) for k in
                ('step','epoch','best_val_cer','global_avg_losses','global_max_losses','log_steps')}

    def load_state_dict(self, d):
        for k, v in d.items(): setattr(self, k, v)


# ============================================================================
# CHECKPOINT MANAGER  (unchanged)
# ============================================================================

class ASRCheckpointManager:
    def __init__(self, checkpoint_dir, model_parts, optimizer, scheduler, train_state, job_config):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_parts = model_parts; self.optimizer = optimizer
        self.scheduler = scheduler; self.train_state = train_state
        self.job_config = job_config
        self.keep_latest_k = getattr(job_config.checkpoint, 'keep_latest_k', 0)

    def save(self, step, force=False, is_best=False):
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        ckpt = {
            'step': step,
            'model_state_dicts':   [m.state_dict() for m in self.model_parts],
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_state':          self.train_state.state_dict(),
            'job_config':           self.job_config.to_dict(),
        }
        if force or step % self.job_config.checkpoint.interval == 0:
            path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"
            torch.save(ckpt, path)
            titan_logger.info(f"Saved checkpoint to {path}")
        if is_best:
            torch.save(ckpt, self.checkpoint_dir / "best_model.pt")
        if self.keep_latest_k > 0:
            for old in sorted(self.checkpoint_dir.glob("checkpoint_step_*.pt"))[:-self.keep_latest_k]:
                old.unlink()

    def load(self, step=-1):
        if step == -1:
            ckpts = sorted(self.checkpoint_dir.glob("checkpoint_step_*.pt"))
            if not ckpts:
                titan_logger.info("No checkpoint found, starting from scratch"); return 0
            path = ckpts[-1]
        else:
            path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"
        if not path.exists():
            titan_logger.warning(f"Checkpoint {path} not found"); return 0
        titan_logger.info(f"Loading checkpoint from {path}")
        ckpt = torch.load(path, map_location='cpu')
        for i, sd in enumerate(ckpt['model_state_dicts']):
            self.model_parts[i].load_state_dict(sd)
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if self.scheduler and ckpt.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.train_state.load_state_dict(ckpt['train_state'])
        return self.train_state.step


# ============================================================================
# DISTRIBUTED SETUP  (unchanged)
# ============================================================================

def setup_distributed(job_config):
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    ft_manager = init_ft_manager(job_config)
    kwargs = dict(
        dp_shard=job_config.training.data_parallel_shard_degree,
        dp_replicate=job_config.training.data_parallel_replicate_degree,
        cp=job_config.experimental.context_parallel_degree,
        tp=job_config.training.tensor_parallel_degree,
        pp=job_config.experimental.pipeline_parallel_degree,
        world_size=world_size,
        enable_loss_parallel=not job_config.training.disable_loss_parallel,
    )
    if ft_manager.enabled:
        parallel_dims = FTParallelDims(**kwargs, ft_manager=ft_manager)
    else:
        parallel_dims = ParallelDims(**kwargs)
    return local_rank, parallel_dims.build_mesh(device_type="cuda"), parallel_dims, ft_manager


# ============================================================================
# TRAIN STEP  (unchanged)
# ============================================================================

# def train_step(model, batch, optimizer, scheduler, grad_accum_steps, max_grad_norm, device, scaler=None):
#     inputs         = batch['features'].to(device, dtype=torch.float32).transpose(1, 2)
#     labels         = batch['targets'].to(device, dtype=torch.long)
#     input_lengths  = batch['feature_lengths'].to(device, dtype=torch.long)
#     target_lengths = batch['target_lengths'].to(device, dtype=torch.long)
#     total_loss = torch.tensor(0.0, device=device)
#     last_output = None
#     for _ in range(grad_accum_steps):
#         if scaler:
#             with torch.cuda.amp.autocast(dtype=torch.bfloat16):
#                 loss = model(input_ids=inputs, labels=labels,
#                              input_lengths=input_lengths, target_lengths=target_lengths).loss / grad_accum_steps
#             # torch.autograd.set_detect_anomaly(True)
#             last_output = loss
#             print("Loss before backward:", loss.item())    
#             scaler.scale(loss).backward()
#         else:
#             loss = model(input_ids=inputs, labels=labels,
#                          input_lengths=input_lengths, target_lengths=target_lengths).loss / grad_accum_steps
#             loss.backward()
#         total_loss += loss.detach()

#     if scaler: scaler.unscale_(optimizer)
#     grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
#     if scaler: scaler.step(optimizer); scaler.update()
#     else:      optimizer.step()
#     if scheduler: scheduler.step()
#     optimizer.zero_grad()
#     act_sparsities = getattr(last_output, 'all_sparsities', [])
    
#     return total_loss, {
#         'loss': total_loss.item(), 'grad_norm': grad_norm.item(),
#         'lr': scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr'],
#         'activation_sparsities': act_sparsities,
#     }




def train_step(model, batch, optimizer, scheduler, grad_accum_steps, max_grad_norm,
               device, scaler=None, global_step=0, total_steps=1,
               reg_start_value=0.0, reg_target_value=0.0):

    inputs         = batch['features'].to(device, dtype=torch.float32).transpose(1, 2)
    labels         = batch['targets'].to(device, dtype=torch.long)
    input_lengths  = batch['feature_lengths'].to(device, dtype=torch.long)
    target_lengths = batch['target_lengths'].to(device, dtype=torch.long)

    # Compute reg lambda for this step
    reg_lambda = Reg_Schedular(global_step, total_steps, reg_start_value, reg_target_value)

    total_loss     = torch.tensor(0.0, device=device)
    total_ctc_loss = torch.tensor(0.0, device=device)
    total_reg_loss = torch.tensor(0.0, device=device)
    last_output    = None

    for _ in range(grad_accum_steps):
        if scaler:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                output = model(input_ids=inputs, labels=labels,
                               input_lengths=input_lengths,
                               target_lengths=target_lengths)
                ctc_loss = output.loss / grad_accum_steps
                reg_term = getattr(output, 'regularization_term', None)

                if reg_term is not None and reg_lambda > 0.0:
                    reg_loss = (reg_lambda * reg_term) / grad_accum_steps
                    loss = ctc_loss + reg_loss
                else:
                    reg_loss = torch.tensor(0.0, device=device)
                    loss = ctc_loss

            last_output = output
            scaler.scale(loss).backward()
        else:
            output = model(input_ids=inputs, labels=labels,
                           input_lengths=input_lengths,
                           target_lengths=target_lengths)
            ctc_loss = output.loss / grad_accum_steps
            reg_term = output.regularization_term

            if reg_term is not None and reg_lambda > 0.0:
                reg_loss = (reg_lambda * reg_term) / grad_accum_steps
                loss = ctc_loss + reg_loss
            else:
                reg_loss = torch.tensor(0.0, device=device)
                loss = ctc_loss

            last_output = output
            loss.backward()

        total_loss     += loss.detach()
        total_ctc_loss += ctc_loss.detach()
        total_reg_loss += reg_loss.detach()

    if scaler: scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    if scaler: scaler.step(optimizer); scaler.update()
    else:      optimizer.step()
    if scheduler: scheduler.step()
    optimizer.zero_grad()

    act_sparsities = getattr(last_output, 'all_sparsities', [])
    
    if dist.is_initialized():
        dist.all_reduce(total_loss, op=dist.ReduceOp.AVG)
        dist.all_reduce(total_ctc_loss, op=dist.ReduceOp.AVG)
        dist.all_reduce(total_reg_loss, op=dist.ReduceOp.AVG)
    
    return total_loss, {
        'loss':              total_loss.item(),
        'loss_ctc':          total_ctc_loss.item(),
        'loss_reg':          total_reg_loss.item(),
        'loss_reg_raw':      reg_term.item() if (reg_term is not None) else 0.0,
        'reg_lambda':        reg_lambda,
        'grad_norm':         grad_norm.item(),
        'lr':                scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr'],
        'activation_sparsities': act_sparsities,
    }

# ============================================================================
# EVALUATE  (unchanged)
# ============================================================================

@torch.no_grad()
def evaluate(model, dataloader, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_sparsities = []

    for i, batch in enumerate(dataloader):
        if max_batches and i >= max_batches:
            break

        inputs = batch['features'].to(device, dtype=torch.float32).transpose(1, 2)
        labels = batch['targets'].to(device, dtype=torch.long)
        input_lengths = batch['feature_lengths'].to(device, dtype=torch.long)
        target_lengths = batch['target_lengths'].to(device, dtype=torch.long)

        outputs = model(input_ids=inputs, labels=labels,
                        input_lengths=input_lengths,
                        target_lengths=target_lengths)
        total_loss += outputs.loss.item()

        # collect sparsity
        if outputs.all_sparsities:
            all_sparsities.extend(outputs.all_sparsities)

        preds = (model.decode(inputs, input_lengths, use_beam_search=False)
                 if hasattr(model, 'decode')
                 else decode_predictions_ctc(outputs.logits, input_lengths))

        start = 0
        for j in range(len(preds)):
            t = target_lengths[j].item()
            all_targets.append(labels[start:start + t].tolist())
            start += t
        all_preds.extend(preds)

    metrics = calculate_asr_metrics(all_preds, all_targets, dataloader.idx_to_char)
    metrics['loss'] = total_loss / len(dataloader)

    # add sparsity metrics if available
    if all_sparsities:
        metrics['act_sparsity_mean'] = sum(all_sparsities) / len(all_sparsities)
        metrics['act_sparsity_min'] = min(all_sparsities)
        metrics['act_sparsity_max'] = max(all_sparsities)

    model.train()
    return metrics

# ============================================================================
# MAIN
# ============================================================================

def main(job_config: JobConfig):
    local_rank, world_mesh, parallel_dims, ft_manager = setup_distributed(job_config)
    device     = torch.device(f"cuda:{local_rank}")
    model_name = getattr(job_config.model, 'name', 'hgrn_asr')
    titan_logger.info(f"Training ASR model: {model_name}")

    # --- Dataloaders ---
    train_loader, val_loader, test_loader = build_asr_dataloaders(job_config)

    n_classes   = train_loader.n_classes
    input_dim   = train_loader.input_dim
    seq_len     = train_loader.seq_len if train_loader.seq_len > 0 else 1000
    idx_to_char = train_loader.idx_to_char
    titan_logger.info(f"Dataset: {n_classes} classes, input_dim={input_dim}, seq_len={seq_len}")

    # CHANGE 2 of 2: build model via registry.
    # Before: train_spec.build_model_fn(config_path=..., input_dim=..., vocab_size=..., ...)
    # After:  build_model(model_name, job_config, metadata)
    #
    # To use a new model: add @register_asr_model("name") in asr_model_registry.py
    # and set model.name = "name" in your job config. Nothing here ever changes.
    metadata = {
        "n_classes":   n_classes,
        "input_dim":   input_dim,
        "seq_len":     seq_len,
        "char_to_idx": train_loader.char_to_idx,
        "idx_to_char": idx_to_char,
    }
    model = build_model(model_name, job_config, metadata)

    # 2. Wrap with FSDP (while still on meta)
    # if parallel_dims.world_size > 1:
    #     parallelize_fla(model, world_mesh, parallel_dims, job_config)

    # 3. Materialize weights onto real CUDA memory
#     model.to_empty(device=device)

# # 4. Initialize weights — THIS is what was missing before
#     with torch.no_grad():
#         model.post_init()

    model.train()

    optimizer = build_asr_optimizer([model], job_config)
    scheduler = build_asr_scheduler(optimizer, job_config)
    scaler    = torch.cuda.amp.GradScaler() if job_config.training.mixed_precision_param == "bfloat16" else None

    train_state = ASRTrainState()
    ckpt_dir    = os.path.join(job_config.job.dump_folder, job_config.checkpoint.folder)
    ckpt_mgr    = ASRCheckpointManager(ckpt_dir, [model], optimizer, scheduler, train_state, job_config)

    if job_config.checkpoint.load_step >= 0:
        train_state.step = ckpt_mgr.load(job_config.checkpoint.load_step)

    logger = TimeSeriesLogger(enable_colors=not job_config.metrics.disable_color_printing, device=str(device))

    if HAS_WANDB and job_config.metrics.enable_wandb and local_rank == 0:
        wandb.init(project=getattr(job_config, 'wandb_project', 'fla-asr'),
                   name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                   config=job_config.to_dict())

    train_iter  = iter(train_loader)
    global_step = train_state.step
    gc_handler  = utils.GarbageCollection(gc_freq=job_config.training.gc_freq)
    titan_logger.info(f"Starting training from step {global_step}")

    while global_step < job_config.training.steps:
        global_step += 1
        train_state.step = global_step
        gc_handler.run(global_step)

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
            train_state.epoch += 1

        loss, metrics = train_step(
            model, batch, optimizer, scheduler,
            job_config.training.gradient_accumulation_steps,
            job_config.training.max_norm,
            device, scaler,
            global_step=global_step,
            total_steps=job_config.training.steps,
            reg_start_value=getattr(job_config.training, 'reg_start_value', 0.0),
            reg_target_value=getattr(job_config.training, 'reg_target_value', 0.0),
        )

        # In main() - update wandb logging block
        if global_step % job_config.metrics.log_freq == 0:
            if HAS_WANDB and job_config.metrics.enable_wandb and local_rank == 0:
                wandb_log = {
                    'train/loss':         metrics['loss'],
                    'train/loss_ctc':     metrics['loss_ctc'],
                    'train/loss_reg':     metrics['loss_reg'],
                    'train/loss_reg_raw': metrics['loss_reg_raw'],
                    'train/reg_lambda':   metrics['reg_lambda'],
                    'train/lr':           metrics['lr'],
                    'train/grad_norm':    metrics['grad_norm'],
                    'train/step':         global_step,
                }
                if metrics['activation_sparsities']:
                    sparsities = metrics['activation_sparsities']
                    wandb_log.update({
                        'sparsity/act_mean': sum(sparsities) / len(sparsities),
                        'sparsity/act_min':  min(sparsities),
                        'sparsity/act_max':  max(sparsities),
                    })
                wandb.log(wandb_log)

        if global_step % (job_config.metrics.log_freq * 10) == 0:
            val_metrics = evaluate(model, val_loader, device, max_batches=50)
            titan_logger.info(f"Validation at step {global_step} | Loss: {val_metrics['loss']:.4f} | "
                            f"CER: {val_metrics['cer']:.4f} | WER: {val_metrics['wer']:.4f}")
            is_best = val_metrics['cer'] < train_state.best_val_cer
            if is_best:
                train_state.best_val_cer = val_metrics['cer']

            if HAS_WANDB and job_config.metrics.enable_wandb and local_rank == 0:
                wandb_log = {
                    'val/loss': val_metrics['loss'],
                    'val/cer': val_metrics['cer'],
                    'val/wer': val_metrics['wer'],
                    'val/step': global_step,
                }
                if 'act_sparsity_mean' in val_metrics:
                    wandb_log.update({
                        'val/sparsity/act_mean': val_metrics['act_sparsity_mean'],
                        'val/sparsity/act_min': val_metrics['act_sparsity_min'],
                        'val/sparsity/act_max': val_metrics['act_sparsity_max'],
                    })
                wandb.log(wandb_log)
            ckpt_mgr.save(global_step, is_best=is_best)

    titan_logger.info("Running final evaluation on test set...")
    test_metrics = evaluate(model, test_loader, device)
    # logger.info(f"Test Results | Loss: {test_metrics['loss']:.4f} | "
    #             f"CER: {test_metrics['cer']:.4f} | WER: {test_metrics['wer']:.4f} | "
    #             f"Exact Match: {test_metrics['exact_match']:.4f}")
    if HAS_WANDB and job_config.metrics.enable_wandb and local_rank == 0:
        wandb.log({'test/loss': test_metrics['loss'], 'test/cer': test_metrics['cer'],
                   'test/wer': test_metrics['wer'], 'test/exact_match': test_metrics['exact_match']})
        wandb.finish()

    titan_logger.info("Training completed!")


# ============================================================================
# ENTRY POINT
# ============================================================================

def parse_args_and_run():
    config = JobConfig()
    config.parse_args()
    if not hasattr(config.training, 'dataset'):  config.training.dataset  = 'librispeech'
    if not hasattr(config.training, 'num_mfcc'): config.training.num_mfcc = 80
    main(config)


if __name__ == "__main__":
    parse_args_and_run()