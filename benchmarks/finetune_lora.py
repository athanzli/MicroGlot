"""LoRA fine-tuning on the GUE tasks, on one GPU.

    CUDA_VISIBLE_DEVICES=0 python finetune_lora.py --model MicroGlot --task EMP/H3 --gue GUE \
        --species "Saccharomyces cerevisiae"
    CUDA_VISIBLE_DEVICES=0 python finetune_lora.py --model MicroGlot-plain --task virus/covid --gue GUE
    CUDA_VISIBLE_DEVICES=0 python finetune_lora.py --model DNABERT-2 --task fungi/species_20 --gue GUE --batch 8 --accum 4

--gue is the unzipped GUE_v2 archive of DNABERT-2 (the first GUE archive has no fungi/species_20 or
virus/species_40); every task folder holds its own train.csv, dev.csv and test.csv. The paper used MicroGlot with the
yeast species on the epigenetic-mark tasks and MicroGlot-plain on the taxonomic tasks.

LoRA (r 16, alpha 32, dropout 0.05) on every linear layer; the head standardises the pooled embedding with running
statistics, then Linear(1024), ReLU, Linear. Each model reads windows of its default context length: a longer sequence
is split into non-overlapping windows that are trained on as separate examples, and their class probabilities are
averaged per sequence at test time. Effective batch 32; the head's running statistics are updated once per
micro-batch, so the split matters. The paper used 32 x 1 except: 8 x 4 on virus/species_40 and fungi/species_20
(NT-2.5B-multi-species 4 x 8 on both, NT-v2-500M 4 x 8 on fungi/species_20), NT-2.5B-multi-species 16 x 2 on
virus/covid, Evo2-7B-base 8 x 4, and Evo2-7B-base 8 x 1 on each of 4 GPUs on virus/covid and fungi/species_20
(multi-GPU training is not part of this script), with --head-norm-floor 0.01 on virus/covid.
"""
import argparse
import json
import os
import re
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from transformers import (AutoModel, AutoModelForMaskedLM, AutoTokenizer, EarlyStoppingCallback, Trainer,
                          TrainerCallback, TrainingArguments)

ENCODERS = {
    "ProkBERT-mini-long": "neuralbioinfo/prokbert-mini-long",
    "DNABERT-2": "zhihan1996/DNABERT-2-117M",
    "DNABERT-S": "zhihan1996/DNABERT-S",
    "NT-v2-250M": "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species",
    "NT-v2-500M": "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
    "NT-2.5B-multi-species": "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
}
MODELS = ["MicroGlot", "MicroGlot-plain", *ENCODERS, "Evo2-7B-base"]
MACRO_F1_TASKS = ("virus/covid", "virus/species_40", "fungi/species_20")  # the epigenetic marks use MCC


def linear_layers(module):
    """Every nn.Linear of `module`, as an exact-name pattern for LoRA."""
    return "(" + "|".join(re.escape(n) for n, m in module.named_modules() if isinstance(m, nn.Linear)) + ")"


def default_context(tok, config):
    lengths = [getattr(config, "max_position_embeddings", None), tok.model_max_length]
    return min(n for n in lengths if n and n < 10**6)


class Evo2Tokenizer:
    """One token per nucleotide; no special tokens; padding id 0."""
    pad_token_id = 0

    def __init__(self, inner):
        self.inner = inner

    def __call__(self, seq, add_special_tokens=True, truncation=False, max_length=None):
        ids = list(self.inner.tokenize(seq))
        return {"input_ids": ids[:max_length] if truncation and max_length else ids}


class CastLinear(nn.Linear):
    def forward(self, x):
        return super().forward(x.to(self.weight.dtype))


class Evo2Backbone(nn.Module):
    """Evo2 as embedding -> blocks -> norm, trainable with LoRA."""

    def __init__(self, core, checkpointing):
        super().__init__()
        for mod in core.modules():  # weights are loaded under inference_mode: copy them into ordinary tensors
            for n, p in list(mod.named_parameters(recurse=False)):
                setattr(mod, n, nn.Parameter(p.detach().clone(), requires_grad=False))
            for n, b in list(mod.named_buffers(recurse=False)):
                if b is not None:
                    mod.register_buffer(n, b.detach().clone())
        for name, mod in list(core.named_modules()):  # TELinear -> nn.Linear, so LoRA reaches the input projections
            if type(mod).__name__ == "TELinear":
                new = CastLinear(mod.in_features, mod.out_features, bias=mod.has_bias,
                                 device=mod.weight.device, dtype=mod.weight.dtype)
                with torch.no_grad():
                    new.weight.copy_(mod.weight)
                    if mod.has_bias and mod.bias is not None:
                        new.bias.copy_(mod.bias)
                parent, child = name.rsplit(".", 1)
                setattr(core.get_submodule(parent), child, new)
        self.core, self.checkpointing = core, checkpointing

    def forward(self, input_ids, attention_mask=None):
        x = self.core.embedding_layer(input_ids)
        for block in self.core.blocks:
            if self.checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    lambda h, b=block: b(h, inference_params=None, padding_mask=None)[0], x, use_reentrant=False)
            else:
                out = block(x, inference_params=None, padding_mask=None)
                x = out[0] if isinstance(out, tuple) else out
        return SimpleNamespace(last_hidden_state=self.core.norm(x))


def load(model, median_len):
    """-> backbone, tokenizer, hidden size, LoRA target modules, context length (tokens)."""
    if model.startswith("MicroGlot"):
        sub = {"subfolder": "plain"} if model == "MicroGlot-plain" else {}
        tok = AutoTokenizer.from_pretrained("athanzli/MicroGlot", trust_remote_code=True, **sub)
        bb = AutoModel.from_pretrained("athanzli/MicroGlot", trust_remote_code=True, torch_dtype=torch.bfloat16, **sub)
        return bb, tok, bb.config.hidden_size, linear_layers(bb), tok.model_max_length
    if model in ENCODERS:
        tok = AutoTokenizer.from_pretrained(ENCODERS[model], trust_remote_code=True)
        try:
            bb = AutoModel.from_pretrained(ENCODERS[model], trust_remote_code=True)
        except ValueError:  # NT: its remote config maps only the masked-LM class
            bb = AutoModelForMaskedLM.from_pretrained(ENCODERS[model], trust_remote_code=True).base_model
        if model.startswith("DNABERT"):  # PyTorch attention: the bundled Triton kernel fails on current Triton
            for name, mod in list(sys.modules.items()):
                if name.endswith("bert_layers"):
                    mod.flash_attn_qkvpacked_func = None
        return bb, tok, bb.config.hidden_size, "all-linear", default_context(tok, bb.config)
    if model == "Evo2-7B-base":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from baselines import _evo2_disable_fp8_input_projections, _patch_evo2_flash_attn
        _patch_evo2_flash_attn()  # PyTorch attention and no FP8, as in the probing extractor
        _evo2_disable_fp8_input_projections(models=("evo2_7b_base",))
        from evo2 import Evo2
        ev = Evo2("evo2_7b_base")
        bb = Evo2Backbone(ev.model, checkpointing=median_len > 2048)
        return bb, Evo2Tokenizer(ev.tokenizer), 4096, ["l1", "l2", "l3", "out_filter_dense", "Wqkv", "out_proj",
                                                         "projections"], 8192
    raise ValueError(f"--model must be one of {MODELS}")


def windows(seq, tok, context):
    """Token ids of `seq` as one window, or as non-overlapping windows of the context ([BOS]..[EOS] if the
    tokenizer has both)."""
    ids = tok(seq, add_special_tokens=False)["input_ids"]
    if len(ids) <= context:
        return [tok(seq, truncation=True, max_length=context)["input_ids"]]
    bos, eos = getattr(tok, "bos_token_id", None), getattr(tok, "eos_token_id", None)
    wrap = bos is not None and eos is not None
    step = context - 2 if wrap else context
    return [[bos, *ids[i:i + step], eos] if wrap else ids[i:i + step] for i in range(0, len(ids), step)]


class WindowData(torch.utils.data.Dataset):
    def __init__(self, df, tok, context):
        self.items = [(w, int(y), i) for i, (s, y) in enumerate(zip(df["sequence"].astype(str), df["label"]))
                      for w in windows(s, tok, context)]
        self.parent = np.array([p for _, _, p in self.items])
        self.labels = df["label"].astype(int).to_numpy()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ids, y, _ = self.items[i]
        return {"input_ids": ids, "labels": y}


def collate(pad_id):
    def fn(batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for j, b in enumerate(batch):
            ids[j, :len(b["input_ids"])] = torch.tensor(b["input_ids"])
            mask[j, :len(b["input_ids"])] = 1
        return {"input_ids": ids, "attention_mask": mask, "labels": torch.tensor([b["labels"] for b in batch])}
    return fn


class TrackStd(nn.Module):
    """Standardises each feature with running (EMA) statistics, the same in training and evaluation."""

    def __init__(self, dim, momentum=0.01, floor=1e-3):
        super().__init__()
        self.momentum, self.floor = momentum, floor
        self.register_buffer("mu", torch.zeros(dim))
        self.register_buffer("sd", torch.ones(dim))
        self.register_buffer("n", torch.zeros((), dtype=torch.long))

    def forward(self, x):
        if self.training and x.shape[0] >= 2:
            with torch.no_grad():
                self.n += 1
                m = max(1.0 / float(self.n), self.momentum)
                self.mu.mul_(1 - m).add_(m * x.detach().float().mean(0))
                self.sd.mul_(1 - m).add_(m * x.detach().float().std(0).clamp(min=self.floor))
        return (x - self.mu) / self.sd


class Classifier(nn.Module):
    def __init__(self, backbone, hidden, n_classes, species_id=None, floor=1e-3):
        super().__init__()
        self.backbone, self.species_id = backbone, species_id
        self.head = nn.Sequential(TrackStd(hidden, floor=floor), nn.Linear(hidden, 1024), nn.ReLU(),
                                  nn.Linear(1024, n_classes))

    def forward(self, input_ids, attention_mask, labels=None, num_items_in_batch=None, **kwargs):
        species = {}
        if self.species_id is not None:
            species["species_ids"] = torch.full((len(input_ids),), self.species_id, device=input_ids.device)
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **species)
        h = out[0] if isinstance(out, tuple) else out.last_hidden_state  # DNABERT-2/-S return a tuple
        # mean over every token except padding (MicroGlot's species token is not among its outputs)
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1, dtype=torch.float32) / m.sum(1, dtype=torch.float32).clamp(min=1)
        with torch.autocast(device_type=pooled.device.type, enabled=False):
            logits = self.head(pooled.float())
        loss = None
        if labels is not None:  # Trainer passes the example count of the whole accumulation group
            loss = nn.functional.cross_entropy(logits, labels, reduction="sum" if num_items_in_batch else "mean")
            if num_items_in_batch:
                loss = loss / num_items_in_batch
        return {"loss": loss, "logits": logits}


def scores(prob, y):
    p = prob.argmax(-1)
    return {"acc": accuracy_score(y, p), "mcc": matthews_corrcoef(y, p),
            "macro_f1": f1_score(y, p, average="macro", zero_division=0)}


def per_sequence(logits, data, n_classes):
    """Class probabilities averaged over each sequence's windows."""
    z = np.asarray(logits[0] if isinstance(logits, tuple) else logits, dtype=np.float64)
    prob = np.exp(z - z.max(-1, keepdims=True))
    prob /= prob.sum(-1, keepdims=True)
    out = np.zeros((len(data.labels), n_classes))
    np.add.at(out, data.parent, prob)
    return out / np.bincount(data.parent, minlength=len(data.labels))[:, None]


class TestEachEpoch(TrainerCallback):
    def __init__(self, data, n_classes):
        self.data, self.n_classes, self.trainer, self.rows = data, n_classes, None, []

    def on_epoch_end(self, args, state, control, **kwargs):
        logits = self.trainer.predict(self.data).predictions
        self.rows.append({"epoch": state.epoch, **scores(per_sequence(logits, self.data, self.n_classes),
                                                         self.data.labels)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--task", required=True, help="GUE task folder, e.g. EMP/H3 or fungi/species_20")
    ap.add_argument("--gue", required=True, help="the GUE folder")
    ap.add_argument("--species", default=None, help="MicroGlot only: the species of every sequence")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=32, help="batch per step; batch x accum = 32 in the paper")
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--head-norm-floor", type=float, default=1e-3, help="lower bound of the head's running std")
    args = ap.parse_args()
    args.task = args.task.strip("/")
    if torch.cuda.device_count() > 1:
        sys.exit("Run on one GPU, e.g. CUDA_VISIBLE_DEVICES=0: with several, transformers would split every batch")
    if (args.model == "MicroGlot") != (args.species is not None):
        sys.exit("--species is required for MicroGlot (use MicroGlot-plain without a species) and only for it")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    split = {s: pd.read_csv(os.path.join(args.gue, args.task, f"{s}.csv")) for s in ("train", "dev", "test")}
    n_classes = int(pd.concat(split.values())["label"].nunique())
    median_len = int(split["train"]["sequence"].astype(str).str.len().median())

    backbone, tok, hidden, targets, context = load(args.model, median_len)
    backbone = get_peft_model(backbone, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                                                   target_modules=targets))
    if args.grad_ckpt:
        try:
            backbone.enable_input_require_grads()
            backbone.gradient_checkpointing_enable()
        except (AttributeError, ValueError) as e:  # DNABERT-2/-S and NT-v2 do not support it; Evo2 checkpoints itself
            print(f"gradient checkpointing unavailable for {args.model}: {e}", flush=True)
    species_id = tok.convert_species_to_ids(args.species) if args.species else None
    model = Classifier(backbone, hidden, n_classes, species_id, args.head_norm_floor)
    data = {s: WindowData(df, tok, context) for s, df in split.items()}
    print(f"{args.model} / {args.task}: context {context} tokens, {n_classes} classes, "
          f"{len(data['train'])} training windows", flush=True)

    out = os.path.join(args.out, f"{args.model}__{args.task.replace('/', '_')}")
    targs = TrainingArguments(
        output_dir=out, seed=args.seed, num_train_epochs=args.epochs, per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum, per_device_eval_batch_size=max(8, args.batch), learning_rate=1e-4,
        lr_scheduler_type="cosine_with_min_lr", lr_scheduler_kwargs={"min_lr": 1e-5}, warmup_steps=500,
        weight_decay=0.01, optim="adamw_torch", max_grad_norm=1.0, bf16=True, eval_strategy="epoch",
        save_strategy="epoch", save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, logging_steps=50, report_to=[], remove_unused_columns=False,
        save_safetensors=False)
    test_each_epoch = TestEachEpoch(data["test"], n_classes)
    trainer = Trainer(model=model, args=targs, train_dataset=data["train"], eval_dataset=data["dev"],
                      data_collator=collate(tok.pad_token_id or 0),
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=3), test_each_epoch])
    test_each_epoch.trainer = trainer
    trainer.train()

    test = scores(per_sequence(trainer.predict(data["test"]).predictions, data["test"], n_classes),
                  data["test"].labels)
    primary = "macro_f1" if args.task in MACRO_F1_TASKS else "mcc"
    result = {"model": args.model, "task": args.task, "seed": args.seed, "primary_metric": primary,
              "test_primary": test[primary], **{f"test_{k}": v for k, v in test.items()},
              "best_val_loss": trainer.state.best_metric, "test_per_epoch": test_each_epoch.rows}
    json.dump(result, open(os.path.join(out, "result.json"), "w"), indent=1)
    print(f"{args.model} / {args.task}: test {primary} {test[primary]:.4f}", flush=True)


if __name__ == "__main__":
    main()
