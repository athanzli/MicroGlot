"""Layer-wise probing (Methods, Probing): a mean-pooled embedding from every layer, then an MLP probe per layer.

    python layer_sweep.py --model MicroGlot-plain --data task.csv --task-type classification
    python layer_sweep.py --model MicroGlot --data traits.csv --task-type classification
    python layer_sweep.py --model DNABERT-2 --data growth.csv --task-type regression

The CSV has one row per sequence with `sequence` and `label`, and optionally `assembly` (rows sharing it are the
contigs of one assembly; their embeddings are averaged), `species` (MicroGlot; an empty cell is inferred by the
Species-encoder) and `split` (train/dev/test: one probe per seed trained on train and stopped early on dev; without it,
10-fold cross-validation with an 8:1:1 train/validation/test split). Seeds 0, 1 and 2. The paper used
--probe-batch 2048 on the marine, plant & synthetic species task (64 for Evo2-7B-base) and 64 elsewhere. Writes every
fold's scores to --out and prints the mean test score per layer, the mean over layers 1 to L, and the test score of the
layer selected on validation.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baselines  # noqa: E402


class MicroGlotLayers:
    """Every hidden state of MicroGlot or MicroGlot-plain, mean-pooled over all tokens except padding
    (the species token is not among MicroGlot's outputs)."""
    supports_chunk_batching = True

    def __init__(self, plain, species):
        sub = {"subfolder": "plain"} if plain else {}
        self.tokenizer = AutoTokenizer.from_pretrained("athanzli/MicroGlot", trust_remote_code=True, **sub)
        self.model = AutoModel.from_pretrained("athanzli/MicroGlot", trust_remote_code=True,
                                               torch_dtype=torch.bfloat16, **sub).cuda().eval()
        if not plain and (species is None or (species < 0).any()):
            self.model.load_species_encoder()
        self.context = self.tokenizer.model_max_length
        self.layers = list(range(self.model.config.num_hidden_layers + 1))

    @torch.no_grad()
    def __call__(self, input_ids, attention_mask, species_id=None):
        extra = {} if species_id is None else {"species_ids": torch.full((len(input_ids),), species_id).cuda()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hs = self.model(input_ids=input_ids.cuda(), attention_mask=attention_mask.cuda(), output_hidden_states=True,
                            **extra).hidden_states
        m = attention_mask.cuda().unsqueeze(-1).float()
        return torch.stack([(h * m).sum(1) / m.sum(1) for h in hs], 1).float().cpu()


class BaselineLayers:
    def __init__(self, name):
        self.ext = baselines.get_extractor(name)
        self.tokenizer = self.ext.get_tokenizer()
        self.context = baselines.MODEL_CONFIGS[name]["max_length"]
        self.layers = list(range(self.ext.model.config.num_hidden_layers + 1))
        self.supports_chunk_batching = getattr(self.ext, "supports_chunk_batching", False)

    def __call__(self, input_ids, attention_mask, species_id=None):
        return self.ext.encode_multilayer(input_ids, attention_mask, self.layers).float().cpu()


def token_ids(seq, tok, **kw):
    ids = tok(seq, **kw)["input_ids"]
    return ids[0].tolist() if torch.is_tensor(ids) else list(ids)


@torch.no_grad()
def embed(seq, enc, species_id=None):
    """[layers, hidden] for one sequence: non-overlapping windows of the context ([BOS]..[EOS] if the tokenizer has
    both), each mean-pooled per layer, then averaged."""
    tok, ctx = enc.tokenizer, enc.context
    ids = token_ids(seq, tok, add_special_tokens=False)
    if len(ids) <= ctx:
        chunks = [token_ids(seq, tok, truncation=True, max_length=ctx)]
    else:
        bos, eos = getattr(tok, "bos_token_id", None), getattr(tok, "eos_token_id", None)
        wrap = bos is not None and eos is not None
        step = ctx - 2 if wrap else ctx
        chunks = [[bos, *ids[i:i + step], eos] if wrap else ids[i:i + step] for i in range(0, len(ids), step)]
    pad = getattr(tok, "pad_token_id", None) or 0
    per_batch = 24 if enc.supports_chunk_batching else 1
    out = []
    for b in range(0, len(chunks), per_batch):
        batch = chunks[b:b + per_batch]
        n = max(map(len, batch))
        input_ids = torch.full((len(batch), n), pad, dtype=torch.long)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for r, c in enumerate(batch):
            input_ids[r, :len(c)] = torch.tensor(c)
            mask[r, :len(c)] = 1
        out.append(enc(input_ids, mask, species_id))
    return torch.cat(out).mean(0)


class MLP(nn.Module):
    def __init__(self, d, n_out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 1024), nn.ReLU(), nn.Linear(1024, n_out))

    def forward(self, x):
        return self.net(x)


def train_probe(x, y, xv, yv, n_out, classification, seed, batch=64):
    """AdamW (weight decay 0.01; lr 1e-4 at batch 64, scaled by sqrt(batch / 64)), gradient clipping 1.0, at most 100
    epochs, early stopping after 5 epochs without a lower validation loss; returns the best epoch's weights."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = MLP(x.shape[1], n_out).cuda()
    loss_fn = nn.CrossEntropyLoss() if classification else nn.MSELoss()
    yt = torch.tensor(y, dtype=torch.long if classification else torch.float32)
    yv = torch.tensor(yv, dtype=torch.long if classification else torch.float32).cuda()
    if not classification:
        yt, yv = yt[:, None], yv[:, None]
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(x), yt), shuffle=True,
                                         batch_size=batch)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4 * (batch / 64) ** 0.5, weight_decay=0.01)
    xv = torch.tensor(xv).cuda()
    best, best_state, wait = float("inf"), None, 0
    for _ in range(100):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(model(xb.cuda()), yb.cuda())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            val_loss = float(loss_fn(model(xv).float(), yv))
        if val_loss < best:
            best, best_state, wait = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= 5:
                break
    model.load_state_dict(best_state)
    return model, best


def predict(model, x, classification):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(torch.tensor(x).cuda()).float().cpu()
    return out.argmax(-1).numpy() if classification else out[:, 0].numpy()


def scores(y, p, classification):
    if classification:
        return {"accuracy": accuracy_score(y, p), "macro_f1": f1_score(y, p, average="macro", zero_division=0),
                "mcc": matthews_corrcoef(y, p)}
    return {"spearman": spearmanr(y, p).statistic, "r2": r2_score(y, p), "rmse": float(np.sqrt(np.mean((y - p) ** 2)))}


def fit_and_score(x, y, tr, va, tests, classification, seed, batch, n_out):
    sc = StandardScaler().fit(x[tr])
    model, val_loss = train_probe(sc.transform(x[tr]).astype(np.float32), y[tr], sc.transform(x[va]).astype(np.float32),
                                  y[va], n_out, classification, seed, batch)
    row = {"val_loss": val_loss}
    for split, idx in (("val", va), *tests):
        pred = predict(model, sc.transform(x[idx]).astype(np.float32), classification)
        row.update({f"{split}_{m}": v for m, v in scores(y[idx], pred, classification).items()})
    return row


def probe(x, y, classification, split=None, batch=64, seeds=(0, 1, 2), n_folds=10):
    rows = []
    if split is not None:  # the given train/dev/test split
        idx = {s: np.flatnonzero(split == s) for s in ("train", "dev", "test")}
        n_out = int(y[idx["train"]].max()) + 1 if classification else 1
        for seed in seeds:
            rows.append({"seed": seed, "fold": 1, **fit_and_score(x, y, idx["train"], idx["dev"],
                                                                  [("test", idx["test"])], classification, seed,
                                                                  batch, n_out)})
        return rows
    for seed in seeds:
        folds = (StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(x, y) if classification
                 else KFold(n_folds, shuffle=True, random_state=seed).split(x))
        for k, (train, test) in enumerate(folds):
            strat = y[train] if classification and np.unique(y[train], return_counts=True)[1].min() >= 2 else None
            tr, va = train_test_split(train, test_size=1 / 9, stratify=strat, random_state=seed)
            rows.append({"seed": seed, "fold": k + 1, **fit_and_score(x, y, tr, va, [("test", test)], classification,
                                                                     seed + k, batch,
                                                                     int(y.max()) + 1 if classification else 1)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="MicroGlot, MicroGlot-plain or one of " + ", ".join(baselines.MODEL_CONFIGS))
    ap.add_argument("--data", required=True)
    ap.add_argument("--task-type", required=True, choices=["classification", "regression"])
    ap.add_argument("--probe-batch", type=int, default=64)
    ap.add_argument("--out", default=None, help="CSV of every fold's scores (default: <data>__<model>.csv)")
    args = ap.parse_args()
    out = args.out or f"{os.path.splitext(args.data)[0]}__{args.model}.csv"
    if os.path.abspath(out) == os.path.abspath(args.data):
        sys.exit("--out would overwrite --data")
    df = pd.read_csv(args.data)
    classification = args.task_type == "classification"

    species = None
    if args.model == "MicroGlot":
        tok = AutoTokenizer.from_pretrained("athanzli/MicroGlot", trust_remote_code=True)
        names = df["species"].where(df["species"].notna(), None) if "species" in df else [None] * len(df)
        species = np.array(tok.convert_species_to_ids(list(names), unknown_species="infer"))
    enc = (MicroGlotLayers(args.model == "MicroGlot-plain", species) if args.model.startswith("MicroGlot")
           else BaselineLayers(args.model))
    feats = torch.stack([embed(s, enc, None if species is None else int(species[i]))
                         for i, s in enumerate(df["sequence"].astype(str).str.upper())])
    if "assembly" in df:  # a row without an assembly is an item of its own
        asm = df["assembly"].astype(str).where(df["assembly"].notna(), "row " + df.index.astype(str))
        groups = list(df.groupby(asm, sort=False).indices.values())
        feats = torch.stack([feats[idx].mean(0) for idx in groups])
        df = df.iloc[[idx[0] for idx in groups]]
    y = (pd.factorize(df["label"].astype(str), sort=True)[0] if classification
         else df["label"].to_numpy(dtype=np.float32))

    metric = "macro_f1" if classification else "spearman"
    rows = []
    for layer in range(feats.shape[1]):
        rows += [{"layer": layer, **r} for r in probe(feats[:, layer].numpy(), y, classification,
                                                      df["split"].to_numpy() if "split" in df else None,
                                                      args.probe_batch)]
        print(f"layer {layer}: test {metric} {np.mean([r[f'test_{metric}'] for r in rows if r['layer'] == layer]):.4f}",
              flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(out, index=False)

    per_layer = res[res.layer >= 1].groupby("layer")[[f"test_{metric}", f"val_{metric}"]].mean()
    best = int(per_layer[f"val_{metric}"].idxmax())
    print(f"mean over layers 1-{per_layer.index.max()}: {per_layer[f'test_{metric}'].mean():.4f}; "
          f"layer {best} (selected on validation): {per_layer.loc[best, f'test_{metric}']:.4f}")


if __name__ == "__main__":
    main()
