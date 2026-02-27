#!/usr/bin/env python3
"""
Unified ASR Training Framework - Multi-GPU DDP Version
"""

import os
import gc
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from flame.config_manager import JobConfig
from flame.logging_timeseries import TimeSeriesLogger
from sparse_mamba.custom_dataloaders.librosa import create_librosa_raw_classification_dataset
from flame.asr_registry import build_model

from torchtitan.tools.logging import logger as titan_logger
from torchtitan.tools import utils

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ============================================================================
# DISTRIBUTED SETUP  — replaces the old torchtitan-based setup_distributed
# ============================================================================

def setup_ddp():
    """
    Initialize the process group and pin each process to its own GPU.
    Called once at startup on every rank.

    Returns
    -------
    local_rank  : int  – GPU index on this node (0 … n_gpus_per_node-1)
    global_rank : int  – unique rank across all nodes
    world_size  : int  – total number of processes (= total GPUs)
    device      : torch.device
    """
    dist.init_process_group(backend="nccl")          # NCCL is fastest for GPU-to-GPU comms
    local_rank  = int(os.environ["LOCAL_RANK"])       # set by torchrun automatically
    global_rank = dist.get_rank()
    world_size  = dist.get_world_size()
    torch.cuda.set_device(local_rank)                 # each process owns exactly one GPU
    device = torch.device(f"cuda:{local_rank}")
    return local_rank, global_rank, world_size, device


def cleanup_ddp():
    dist.destroy_process_group()


def is_main_process():
    """True only on rank-0. Use to gate logging, checkpointing, wandb."""
    return dist.get_rank() == 0


# ============================================================================
# SCHEDULER HELPER
# ============================================================================

def Reg_Schedular(step, total_steps, start_value, target_value):
    """S-Curve (cosine) hyperparameter scheduler."""
    if step >= total_steps:
        return target_value
    progress = step / total_steps
    factor = 0.5 * (1 - math.cos(math.pi * progress))
    return start_value + (target_value - start_value) * factor


# ============================================================================
# METRICS AND DECODING
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
        pred   = pred.tolist()   if torch.is_tensor(pred)   else list(pred)
        target = target.tolist() if torch.is_tensor(target) else list(target)

        pc  = [p for p in pred   if p != 0]
        tc_ = [t for t in target if t != 0]
        tce += levenshtein_distance(pc, tc_)
        tc  += len(tc_)

        if idx_to_char:
            pred_text   = "".join([idx_to_char.get(t, "") for t in pc]).replace("▁", " ").strip()
            target_text = "".join([idx_to_char.get(t, "") for t in tc_]).replace("▁", " ").strip()
            pred_words   = pred_text.split()
            target_words = target_text.split()
        else:
            pred_words   = pc
            target_words = tc_

        twe += levenshtein_distance(pred_words, target_words)
        tw  += len(target_words)
        em  += pc == tc_
        ml   = min(len(pc), len(tc_))
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

# def _make_empty_dataset(samples, char_to_idx, idx_to_char, num_mfcc, dataset_name):
#     """
#     Build a dataset object from already-loaded samples.
#     Called on ranks 1-N so they never touch disk.
#     """
#     from sparse_mamba.custom_dataloaders.preprocess_ctc import LibriSpeechASRDataset
#     import types

#     # Create instance without calling __init__ (skips load_dataset)
#     ds = object.__new__(LibriSpeechASRDataset)
#     ds.samples      = samples
#     ds.char_to_idx  = char_to_idx
#     ds.idx_to_char  = idx_to_char
#     ds.num_mfcc     = num_mfcc
#     ds.n_mels       = max(num_mfcc, 40)
#     ds.n_fft        = 400
#     ds.hop_length   = 160
#     ds.win_length   = 400
#     ds.f_min        = 0.0
#     ds.f_max        = 8000
#     ds.use_mfcc     = True
#     ds.dataset_name = dataset_name
#     ds.streaming    = False
#     return ds


# ============================================================================
# DATALOADER  — now returns DistributedSamplers so each GPU sees a unique shard
# ============================================================================

# def build_asr_dataloaders(job_config, world_size: int, global_rank: int):
#     """
#     Build train / val / test loaders with DistributedSampler on the training set.

#     Key change vs old version
#     -------------------------
#     We pass `world_size` and `global_rank` so that DistributedSampler can split
#     the training data evenly.  Each GPU processes 1/world_size of every epoch.
#     Val and test use a regular sampler so rank-0 evaluates on the full set.
#     """
#     dataset        = getattr(job_config.training, 'dataset',       'librispeech')
#     max_samples    = getattr(job_config.training, 'max_samples',   -1)
#     num_mfcc       = getattr(job_config.training, 'num_mfcc',      80)
#     cache_dir      = getattr(job_config.training, 'cache_dir',     './cache')
#     spm_model_path = getattr(job_config.training, 'spm_model_path', None)
#     streaming = getattr(job_config.training, 'streaming', False)
#     print(f" =============================== streaming ============================================: {streaming}")

#     # create_librosa_raw_classification_dataset returns plain DataLoaders.
#     # We rebuild them here with DistributedSampler for the training split.
#     from sparse_mamba.custom_dataloaders.preprocess_ctc import (
#         LibriSpeechASRDataset, collate_fn_ctc
#     )

#     def _make_dataset(split, max_s):
#         return LibriSpeechASRDataset(
#             split=split,
#             max_samples=max_s,
#             num_mfcc=num_mfcc,
#             dataset=dataset,
#             spm_model_path=spm_model_path,
#             streaming=streaming,
#         )

#     # Splits
#     if dataset == "librispeech":
#         train_split, val_split, test_split = "train.360", "validation", "test"
#     else:
#         train_split, val_split, test_split = "train", "validation", "test"

#     train_ds = _make_dataset(train_split, max_samples)
#     val_ds   = _make_dataset(val_split,   max_samples)
#     test_ds  = _make_dataset(test_split,  max_samples)

#     # DistributedSampler splits the training data across GPUs.
#     # shuffle=True means each epoch the data is re-shuffled before splitting.
#     train_sampler = DistributedSampler(
#         train_ds,
#         num_replicas=world_size,
#         rank=global_rank,
#         shuffle=True,
#         drop_last=True,   # avoids unequal batch sizes on the last batch
#     )

#     # Val / test: only evaluated on rank-0, so no distributed sampler needed
#     bsz = job_config.training.batch_size
#     nw  = getattr(job_config.training, 'num_workers', 4)

#     train_loader = torch.utils.data.DataLoader(
#         train_ds,
#         batch_size=bsz,
#         sampler=train_sampler,     # replaces shuffle=True
#         collate_fn=collate_fn_ctc,
#         num_workers=nw,
#         pin_memory=True,
#         drop_last=True,
#     )
#     val_loader = torch.utils.data.DataLoader(
#         val_ds,
#         batch_size=bsz,
#         shuffle=False,
#         collate_fn=collate_fn_ctc,
#         num_workers=nw,
#         pin_memory=True,
#     )
#     test_loader = torch.utils.data.DataLoader(
#         test_ds,
#         batch_size=bsz,
#         shuffle=False,
#         collate_fn=collate_fn_ctc,
#         num_workers=nw,
#         pin_memory=True,
#     )

#     # Attach metadata expected by the training loop
#     for loader in (train_loader, val_loader, test_loader):
#         loader.n_classes   = len(train_ds.char_to_idx)
#         loader.input_dim   = num_mfcc
#         loader.seq_len     = -1
#         loader.char_to_idx = train_ds.char_to_idx
#         loader.idx_to_char = train_ds.idx_to_char

#     return train_loader, val_loader, test_loader, train_sampler

# def build_asr_dataloaders(job_config, world_size: int, global_rank: int):
    
#     dataset        = getattr(job_config.training, 'dataset',       'librispeech')
#     max_samples    = getattr(job_config.training, 'max_samples',   -1)
#     num_mfcc       = getattr(job_config.training, 'num_mfcc',      80)
#     spm_model_path = getattr(job_config.training, 'spm_model_path', None)
#     streaming      = getattr(job_config.training, 'streaming',     False)

#     from sparse_mamba.custom_dataloaders.preprocess_ctc import (
#         LibriSpeechASRDataset, collate_fn_ctc
#     )

#     if dataset == "librispeech":
#         train_split, val_split, test_split = "train.360", "validation", "test"
#     else:
#         train_split, val_split, test_split = "train", "validation", "test"

#     def _make_dataset(split, max_s):
#         return LibriSpeechASRDataset(
#             split=split,
#             max_samples=max_s,
#             num_mfcc=num_mfcc,
#             dataset=dataset,
#             spm_model_path=spm_model_path,
#             streaming=streaming,
#         )

#     # ---------------------------------------------------------------
#     # Only rank 0 loads the dataset, then broadcasts samples to all
#     # other ranks so the data is only read from disk/network once.
#     # ---------------------------------------------------------------
#     if global_rank == 0:
#         train_ds = _make_dataset(train_split, max_samples)
#         # Package what we need to share
#         payload = [train_ds.samples, train_ds.char_to_idx, train_ds.idx_to_char]
#     else:
#         train_ds = None
#         payload  = [None, None, None]

#     # Broadcast from rank 0 to all other ranks
#     dist.broadcast_object_list(payload, src=0)
#     samples, char_to_idx, idx_to_char = payload

#     # All ranks now build a dataset from the shared samples
#     # without loading from disk again
#     if global_rank != 0:
#         train_ds = _make_dataset.__func__  # we just need the empty object
#         # Build a lightweight shell with the broadcasted data
#         train_ds = _make_empty_dataset(
#             samples, char_to_idx, idx_to_char, num_mfcc, dataset
#         )
#     else:
#         # rank 0 already has it, just reassign samples to be safe
#         train_ds.samples = samples

#     # Val/test only on rank 0
#     if global_rank == 0:
#         val_ds  = _make_dataset(val_split,  max_samples)
#         test_ds = _make_dataset(test_split, max_samples)
#     else:
#         val_ds  = None
#         test_ds = None

#     # DistributedSampler splits the shared dataset across GPUs
#     train_sampler = DistributedSampler(
#         train_ds,
#         num_replicas=world_size,
#         rank=global_rank,
#         shuffle=True,
#         drop_last=True,
#     )

#     bsz = job_config.training.batch_size
#     nw  = getattr(job_config.training, 'num_workers', 2)

#     train_loader = torch.utils.data.DataLoader(
#         train_ds,
#         batch_size=bsz,
#         sampler=train_sampler,
#         collate_fn=collate_fn_ctc,
#         num_workers=nw,
#         pin_memory=True,
#         drop_last=True,
#     )
#     val_loader = torch.utils.data.DataLoader(
#         val_ds  if val_ds  is not None else train_ds,  # non-rank-0 won't use this
#         batch_size=bsz, shuffle=False,
#         collate_fn=collate_fn_ctc, num_workers=nw, pin_memory=True,
#     )
#     test_loader = torch.utils.data.DataLoader(
#         test_ds if test_ds is not None else train_ds,
#         batch_size=bsz, shuffle=False,
#         collate_fn=collate_fn_ctc, num_workers=nw, pin_memory=True,
#     )

#     for loader in (train_loader, val_loader, test_loader):
#         loader.n_classes   = len(char_to_idx)
#         loader.input_dim   = num_mfcc
#         loader.seq_len     = -1
#         loader.char_to_idx = char_to_idx
#         loader.idx_to_char = idx_to_char

#     return train_loader, val_loader, test_loader, train_sampler

def build_asr_dataloaders(job_config, world_size: int, global_rank: int):
    dataset        = getattr(job_config.training, 'dataset',       'librispeech')
    max_samples    = getattr(job_config.training, 'max_samples',   -1)
    num_mfcc       = getattr(job_config.training, 'num_mfcc',      80)
    spm_model_path = getattr(job_config.training, 'spm_model_path', None)
    streaming      = getattr(job_config.training, 'streaming',     False)

    from sparse_mamba.custom_dataloaders.preprocess_ctc import (
        LibriSpeechASRDataset, collate_fn_ctc
    )

    if dataset == "librispeech":
        train_split, val_split, test_split = "train.360", "validation", "test"
    else:
        train_split, val_split, test_split = "train", "validation", "test"

    def _make_dataset(split, max_s):
        return LibriSpeechASRDataset(
            split=split, max_samples=max_s, num_mfcc=num_mfcc,
            dataset=dataset, spm_model_path=spm_model_path, streaming=streaming,
        )

    # Every rank loads metadata independently from HF disk cache (fast, tiny RAM)
    # DistributedSampler handles the round-robin sharding across GPUs
    train_ds = _make_dataset(train_split, max_samples)
    val_ds   = _make_dataset(val_split,  max_samples) if global_rank == 0 else None
    test_ds  = _make_dataset(test_split, max_samples) if global_rank == 0 else None

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=global_rank,
        shuffle=True, drop_last=True,
    )

    bsz = job_config.training.batch_size
    nw  = getattr(job_config.training, 'num_workers', 2)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=bsz, sampler=train_sampler,
        collate_fn=collate_fn_ctc, num_workers=nw, pin_memory=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds if val_ds is not None else train_ds,
        batch_size=bsz, shuffle=False,
        collate_fn=collate_fn_ctc, num_workers=nw, pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds if test_ds is not None else train_ds,
        batch_size=bsz, shuffle=False,
        collate_fn=collate_fn_ctc, num_workers=nw, pin_memory=True,
    )

    for loader in (train_loader, val_loader, test_loader):
        loader.n_classes   = len(train_ds.char_to_idx)
        loader.input_dim   = num_mfcc
        loader.seq_len     = -1
        loader.char_to_idx = train_ds.char_to_idx
        loader.idx_to_char = train_ds.idx_to_char

    return train_loader, val_loader, test_loader, train_sampler

# ============================================================================
# OPTIMIZER / SCHEDULER
# ============================================================================

def build_asr_optimizer(model, job_config):
    """Pass the raw model (or model.module) — DDP wrapper is transparent here."""
    params = [p for p in model.parameters() if p.requires_grad]
    name   = getattr(job_config.optimizer, 'name', 'AdamW')
    cls    = {'AdamW': torch.optim.AdamW, 'Adam': torch.optim.Adam}.get(name)
    if cls is None:
        raise ValueError(f"Unknown optimizer: {name}")
    return cls(
        params,
        lr=getattr(job_config.optimizer, 'lr', 3e-3),
        weight_decay=getattr(job_config.optimizer, 'weight_decay', 0.01),
        betas=(getattr(job_config.optimizer, 'beta1', 0.9),
               getattr(job_config.optimizer, 'beta2', 0.95)),
        eps=getattr(job_config.optimizer, 'eps', 1e-8),
    )


def build_asr_scheduler(optimizer, job_config):
    from transformers import get_cosine_schedule_with_warmup
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=job_config.lr_scheduler.warmup_steps,
        num_training_steps=job_config.training.steps,
    )


# ============================================================================
# TRAINING STATE
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
                ('step', 'epoch', 'best_val_cer',
                 'global_avg_losses', 'global_max_losses', 'log_steps')}

    def load_state_dict(self, d):
        for k, v in d.items():
            setattr(self, k, v)


# ============================================================================
# CHECKPOINT MANAGER  — always unwraps DDP before saving
# ============================================================================

class ASRCheckpointManager:
    def __init__(self, checkpoint_dir, model, optimizer, scheduler,
                 train_state, job_config):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model       = model          # may be DDP-wrapped
        self.optimizer   = optimizer
        self.scheduler   = scheduler
        self.train_state = train_state
        self.job_config  = job_config
        self.keep_latest_k = getattr(job_config.checkpoint, 'keep_latest_k', 0)

    def _raw_model(self):
        """Unwrap DDP to get the underlying module for state_dict."""
        return self.model.module if isinstance(self.model, DDP) else self.model

    def save(self, step, force=False, is_best=False):
        # Only rank-0 writes files
        if not is_main_process():
            return
        if not (force or step % self.job_config.checkpoint.interval == 0):
            return

        ckpt = {
            'step':                 step,
            'model_state_dict':     self._raw_model().state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_state':          self.train_state.state_dict(),
        }
        path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"
        torch.save(ckpt, path)
        titan_logger.info(f"[rank 0] Saved checkpoint → {path}")

        if is_best:
            torch.save(ckpt, self.checkpoint_dir / "best_model.pt")

        if self.keep_latest_k > 0:
            for old in sorted(
                self.checkpoint_dir.glob("checkpoint_step_*.pt")
            )[: -self.keep_latest_k]:
                old.unlink()

    def load(self, step=-1):
        """All ranks load the same checkpoint (weights are replicated by DDP)."""
        if step == -1:
            ckpts = sorted(self.checkpoint_dir.glob("checkpoint_step_*.pt"))
            if not ckpts:
                titan_logger.info("No checkpoint found — starting from scratch")
                return 0
            path = ckpts[-1]
        else:
            path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"

        if not path.exists():
            titan_logger.warning(f"Checkpoint {path} not found")
            return 0

        titan_logger.info(f"Loading checkpoint from {path}")
        # map_location ensures weights land on the correct device for this rank
        ckpt = torch.load(path, map_location=f"cuda:{torch.cuda.current_device()}")
        self._raw_model().load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if self.scheduler and ckpt.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.train_state.load_state_dict(ckpt['train_state'])
        return self.train_state.step


# ============================================================================
# TRAIN STEP
# ============================================================================

def train_step(model, batch, optimizer, scheduler, grad_accum_steps,
               max_grad_norm, device, scaler=None,
               global_step=0, total_steps=1,
               reg_start_value=0.0, reg_target_value=0.0):

    inputs         = batch['features'].to(device, dtype=torch.float32).transpose(1, 2)
    labels         = batch['targets'].to(device, dtype=torch.long)
    input_lengths  = batch['feature_lengths'].to(device, dtype=torch.long)
    target_lengths = batch['target_lengths'].to(device, dtype=torch.long)

    reg_lambda = Reg_Schedular(global_step, total_steps, reg_start_value, reg_target_value)

    total_loss     = torch.tensor(0.0, device=device)
    total_ctc_loss = torch.tensor(0.0, device=device)
    total_reg_loss = torch.tensor(0.0, device=device)
    last_output    = None

    for _ in range(grad_accum_steps):
        if scaler:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                output   = model(input_ids=inputs, labels=labels,
                                 input_lengths=input_lengths,
                                 target_lengths=target_lengths)
                ctc_loss = output.loss / grad_accum_steps
                reg_term = getattr(output, 'regularization_term', None)
                reg_loss = (reg_lambda * reg_term / grad_accum_steps
                            if reg_term is not None and reg_lambda > 0.0
                            else torch.tensor(0.0, device=device))
                loss = ctc_loss + reg_loss
            scaler.scale(loss).backward()
        else:
            output   = model(input_ids=inputs, labels=labels,
                             input_lengths=input_lengths,
                             target_lengths=target_lengths)
            ctc_loss = output.loss / grad_accum_steps
            reg_term = getattr(output, 'regularization_term', None)
            reg_loss = (reg_lambda * reg_term / grad_accum_steps
                        if reg_term is not None and reg_lambda > 0.0
                        else torch.tensor(0.0, device=device))
            loss = ctc_loss + reg_loss
            loss.backward()

        last_output    = output
        total_loss     += loss.detach()
        total_ctc_loss += ctc_loss.detach()
        total_reg_loss += reg_loss.detach()

    if scaler:
        scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    if scaler:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    if scheduler:
        scheduler.step()
    optimizer.zero_grad()

    act_sparsities = getattr(last_output, 'all_sparsities', [])

    return total_loss, {
        'loss':              total_loss.item(),
        'loss_ctc':          total_ctc_loss.item(),
        'loss_reg':          total_reg_loss.item(),
        'loss_reg_raw':      reg_term.item() if reg_term is not None else 0.0,
        'reg_lambda':        reg_lambda,
        'grad_norm':         grad_norm.item(),
        'lr':                (scheduler.get_last_lr()[0]
                              if scheduler else optimizer.param_groups[0]['lr']),
        'activation_sparsities': act_sparsities,
    }


# ============================================================================
# EVALUATE  — only called on rank-0
# ============================================================================

@torch.no_grad()
def evaluate(model, dataloader, device, max_batches=None):
    """
    Called only on rank-0. We unwrap DDP so we can call model.decode()
    which is defined on the raw model class, not on the DDP wrapper.
    """
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.eval()

    total_loss    = 0.0
    all_preds     = []
    all_targets   = []
    all_sparsities = []

    for i, batch in enumerate(dataloader):
        if max_batches and i >= max_batches:
            break

        inputs         = batch['features'].to(device, dtype=torch.float32).transpose(1, 2)
        labels         = batch['targets'].to(device, dtype=torch.long)
        input_lengths  = batch['feature_lengths'].to(device, dtype=torch.long)
        target_lengths = batch['target_lengths'].to(device, dtype=torch.long)

        outputs = raw_model(input_ids=inputs, labels=labels,
                            input_lengths=input_lengths,
                            target_lengths=target_lengths)
        total_loss += outputs.loss.item()

        if outputs.all_sparsities:
            all_sparsities.extend(outputs.all_sparsities)

        preds = (raw_model.decode(inputs, input_lengths, use_beam_search=False)
                 if hasattr(raw_model, 'decode')
                 else decode_predictions_ctc(outputs.logits, input_lengths))

        start = 0
        for j in range(len(preds)):
            t = target_lengths[j].item()
            all_targets.append(labels[start:start + t].tolist())
            start += t
        all_preds.extend(preds)

    metrics = calculate_asr_metrics(all_preds, all_targets, dataloader.idx_to_char)
    metrics['loss'] = total_loss / max(len(dataloader), 1)

    if all_sparsities:
        metrics['act_sparsity_mean'] = sum(all_sparsities) / len(all_sparsities)
        metrics['act_sparsity_min']  = min(all_sparsities)
        metrics['act_sparsity_max']  = max(all_sparsities)

    raw_model.train()
    return metrics


# ============================================================================
# MAIN
# ============================================================================

def main(job_config: JobConfig):
    # 1. Boot DDP — must happen before any CUDA calls
    local_rank, global_rank, world_size, device = setup_ddp()

    model_name = getattr(job_config.model, 'name', 'hgrn_asr')
    if is_main_process():
        titan_logger.info(f"Training {model_name} on {world_size} GPU(s)")

    # 2. Dataloaders with DistributedSampler
    train_loader, val_loader, test_loader, train_sampler = \
        build_asr_dataloaders(job_config, world_size, global_rank)

    n_classes   = train_loader.n_classes
    input_dim   = train_loader.input_dim
    idx_to_char = train_loader.idx_to_char

    # 3. Build model (same as before — registry call)
    metadata = {
        "n_classes":   n_classes,
        "input_dim":   input_dim,
        "seq_len":     -1,
        "char_to_idx": train_loader.char_to_idx,
        "idx_to_char": idx_to_char,
    }
    model = build_model(model_name, job_config, metadata)
    # model.gradient_checkpointing_enable() 
    model = model.to(device)

    # 4. Wrap with DDP
    #    find_unused_parameters=False is faster; set True only if you get
    #    "Expected to have finished reduction" errors during training.
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=False)

    # 5. Optimizer and scheduler operate on the DDP wrapper — that's fine
    optimizer = build_asr_optimizer(model, job_config)
    scheduler = build_asr_scheduler(optimizer, job_config)
    scaler    = (torch.cuda.amp.GradScaler()
                 if job_config.training.mixed_precision_param == "bfloat16"
                 else None)

    # 6. Train state and checkpoint manager
    train_state = ASRTrainState()
    ckpt_dir    = os.path.join(job_config.job.dump_folder, job_config.checkpoint.folder)
    ckpt_mgr    = ASRCheckpointManager(ckpt_dir, model, optimizer, scheduler,
                                       train_state, job_config)

    if job_config.checkpoint.load_step >= 0:
        # All ranks must call load() so weights stay in sync
        train_state.step = ckpt_mgr.load(job_config.checkpoint.load_step)

    # 7. WandB — only on rank-0
    if HAS_WANDB and job_config.metrics.enable_wandb and is_main_process():
        wandb.init(
            project=getattr(job_config, 'wandb_project', 'fla-asr'),
            name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=job_config.to_dict(),
        )

    # 8. Training loop
    gc_handler  = utils.GarbageCollection(gc_freq=job_config.training.gc_freq)
    global_step = train_state.step
    train_iter  = iter(train_loader)

    if is_main_process():
        titan_logger.info(f"Starting training from step {global_step}")

    while global_step < job_config.training.steps:
        global_step += 1
        train_state.step = global_step
        gc_handler.run(global_step)

        # Tell the sampler which epoch we're in so it re-shuffles correctly.
        # Without this every epoch has the same data order on every GPU.
        try:
            batch = next(train_iter)
        except StopIteration:
            train_state.epoch += 1
            train_sampler.set_epoch(train_state.epoch)   # ← critical for shuffle
            train_iter = iter(train_loader)
            batch = next(train_iter)

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

        # Logging — only rank-0 prints / logs to wandb
        if global_step % job_config.metrics.log_freq == 0 and is_main_process():
            titan_logger.info(
                f"step={global_step}  loss={metrics['loss']:.4f}  "
                f"ctc={metrics['loss_ctc']:.4f}  "
                f"lr={metrics['lr']:.2e}  grad_norm={metrics['grad_norm']:.3f}"
            )
            if HAS_WANDB and job_config.metrics.enable_wandb:
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
                    sp = metrics['activation_sparsities']
                    wandb_log.update({
                        'sparsity/act_mean': sum(sp) / len(sp),
                        'sparsity/act_min':  min(sp),
                        'sparsity/act_max':  max(sp),
                    })
                wandb.log(wandb_log)

        # Validation — only rank-0 evaluates to avoid redundant computation
        if global_step % (job_config.metrics.log_freq * 10) == 0 and is_main_process():
            val_metrics = evaluate(model, val_loader, device, max_batches=50)
            titan_logger.info(
                f"[val] step={global_step}  loss={val_metrics['loss']:.4f}  "
                f"CER={val_metrics['cer']:.4f}  WER={val_metrics['wer']:.4f}"
            )
            is_best = val_metrics['cer'] < train_state.best_val_cer
            if is_best:
                train_state.best_val_cer = val_metrics['cer']

            if HAS_WANDB and job_config.metrics.enable_wandb:
                wandb.log({'val/loss': val_metrics['loss'],
                           'val/cer':  val_metrics['cer'],
                           'val/wer':  val_metrics['wer'],
                           'val/step': global_step})

            ckpt_mgr.save(global_step, is_best=is_best)

        # All ranks must hit this barrier together before moving to the next step.
        # This prevents rank-0 from racing ahead while others are still in backward.
        dist.barrier()

    # Final test evaluation on rank-0 only
    if is_main_process():
        titan_logger.info("Running final evaluation on test set...")
        test_metrics = evaluate(model, test_loader, device)
        titan_logger.info(
            f"[test] loss={test_metrics['loss']:.4f}  "
            f"CER={test_metrics['cer']:.4f}  WER={test_metrics['wer']:.4f}  "
            f"EM={test_metrics['exact_match']:.4f}"
        )
        if HAS_WANDB and job_config.metrics.enable_wandb:
            wandb.log({'test/loss': test_metrics['loss'],
                       'test/cer':  test_metrics['cer'],
                       'test/wer':  test_metrics['wer']})
            wandb.finish()

    cleanup_ddp()
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