#!/usr/bin/env python3
"""
Generate UCE-compatible protein embeddings for the dog proteome using
ESM2-15B (esm2_t48_15B_UR50D, 5120-dim) via Hugging Face `transformers`.

WHY 15B: UCE's protein-token dimension is 5120. Only esm2_t48_15B_UR50D
produces 5120-dim embeddings. ESM2-3B (2560) and ESM2-650M/ESM1b (1280) are
NOT compatible without retraining UCE. This is a hard requirement.

WHAT IT DOES
  - Reads a peptide FASTA (headers = gene id; ours are ENSCAFG gene_ids).
  - Loads ESM2-15B in fp16 on a single GPU (fits 48 GB, e.g. RunPod A40).
  - Truncates sequences to 1022 residues (ESM2 limit; matches UCE default).
  - Mean-pools the final-layer per-residue representations (excluding the
    <cls>/<eos>/<pad> tokens) -> one 5120-d vector per sequence. This matches
    fair-esm `extract.py --repr_layers 48 --include mean`, which is how the
    UCE protein embeddings were originally produced.
  - Saves a dict {gene_id: FloatTensor[5120]} with torch.save -> the .pt that
    UCE's "Create New Species Files" step consumes.
  - Checkpoints as it goes and can --resume, so a spot/interruptible pod that
    dies mid-run loses nothing.

USAGE
  # 1) smoke test on 50 seqs first (~1 min) to confirm GPU + memory are fine:
  python embed_esm2.py --fasta dog_longest.pep.fa --out dog_esm2.pt --limit 50
  # 2) full run:
  python embed_esm2.py --fasta dog_longest.pep.fa --out dog_esm2_embeddings.pt

KEYING NOTE
  Output is keyed by the FASTA header (ENSCAFG gene_id). For the UCE run,
  key the dog h5ad var_names by the SAME ENSCAFG ids (see next_steps.md).
  Cross-species homology in UCE is carried by the ESM2 embedding space, NOT by
  shared gene symbols, so ENSCAFG-throughout is correct and avoids all
  symbol-collision/renaming problems.
"""
import argparse, os, sys, time
import torch

MODEL_DEFAULT = "facebook/esm2_t48_15B_UR50D"
EXPECT_DIM = 5120


def read_fasta(path):
    ids, seqs, cur, buf = [], [], None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if cur is not None:
                    seqs.append("".join(buf))
                cur = line[1:].strip().split()[0]
                ids.append(cur); buf = []
            else:
                buf.append(line.strip())
    if cur is not None:
        seqs.append("".join(buf))
    return ids, seqs


def make_batches(items, token_budget, max_seqs, max_len):
    """items: list of (id, seq) sorted by length. Greedy dynamic batching by
    total token budget so short seqs pack densely and long seqs run small."""
    batch, cur_tok = [], 0
    for gid, seq in items:
        L = min(len(seq), max_len) + 2  # +cls +eos
        if batch and (cur_tok + L > token_budget or len(batch) >= max_seqs):
            yield batch
            batch, cur_tok = [], 0
        batch.append((gid, seq)); cur_tok += L
    if batch:
        yield batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--max-len", type=int, default=1022, help="residue cap (ESM2 limit)")
    ap.add_argument("--token-budget", type=int, default=4096,
                    help="max tokens per batch; lower if OOM, raise if you have headroom")
    ap.add_argument("--max-seqs", type=int, default=64, help="max sequences per batch")
    ap.add_argument("--checkpoint-every", type=int, default=200, help="save every N batches")
    ap.add_argument("--resume", action="store_true", help="skip ids already in the checkpoint")
    ap.add_argument("--limit", type=int, default=0, help="only embed first N (smoke test)")
    ap.add_argument("--store-dtype", choices=["fp16", "fp32"], default="fp16")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "No CUDA GPU visible."
    dev = "cuda"
    store_dtype = torch.float16 if args.store_dtype == "fp16" else torch.float32
    ckpt = args.out + ".ckpt"

    from transformers import AutoTokenizer, AutoModel

    ids, seqs = read_fasta(args.fasta)
    if args.limit:
        ids, seqs = ids[: args.limit], seqs[: args.limit]
    print(f"[data] {len(ids):,} sequences from {args.fasta}", flush=True)

    emb = {}
    if args.resume and os.path.exists(ckpt):
        emb = torch.load(ckpt, map_location="cpu")
        print(f"[resume] loaded {len(emb):,} embeddings from {ckpt}", flush=True)

    todo = [(g, s) for g, s in zip(ids, seqs) if g not in emb and len(s) > 0]
    todo.sort(key=lambda x: len(x[1]))  # ascending length -> efficient packing
    print(f"[data] {len(todo):,} still to embed "
          f"({sum(1 for _, s in todo if len(s) > args.max_len):,} exceed {args.max_len} aa, truncated)",
          flush=True)
    if not todo:
        print("[done] nothing to do; writing final output.", flush=True)

    print(f"[model] loading {args.model} in fp16 ...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    # load fp16 weights streamed directly onto the GPU (low CPU-RAM footprint)
    try:
        model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float16,
                                          low_cpu_mem_usage=True, device_map={"": 0})
    except Exception as e:
        print(f"[model] device_map load failed ({e}); falling back to CPU->GPU", flush=True)
        model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float16,
                                          low_cpu_mem_usage=True).to(dev)
    model.eval()
    print(f"[model] ready in {time.time()-t0:.0f}s  "
          f"(hidden={model.config.hidden_size})", flush=True)
    assert model.config.hidden_size == EXPECT_DIM, \
        f"hidden {model.config.hidden_size} != {EXPECT_DIM}; wrong model?"

    n_batches = 0
    done = len(emb)
    t_run = time.time()
    for batch in make_batches(todo, args.token_budget, args.max_seqs, args.max_len):
        bids = [b[0] for b in batch]
        bseq = [b[1] for b in batch]
        enc = tok(bseq, return_tensors="pt", padding=True, truncation=True,
                  max_length=args.max_len + 2)
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        h = out.last_hidden_state              # [B, L, 5120] fp16
        am = enc["attention_mask"].clone()     # [B, L]
        am[:, 0] = 0                            # drop <cls>
        lengths = enc["attention_mask"].sum(1) # incl cls+eos
        am[torch.arange(am.size(0)), (lengths - 1).long()] = 0  # drop <eos>
        m = am.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)   # [B, 5120]
        pooled = pooled.to(store_dtype).cpu()
        for gid, vec in zip(bids, pooled):
            emb[gid] = vec
        done += len(bids); n_batches += 1
        if n_batches % 20 == 0:
            rate = done / max(time.time() - t_run, 1e-6)
            print(f"[run] {done:,}/{len(ids):,}  ({rate:.0f} seq/s)  "
                  f"batch={len(bids)}  maxlen={max(len(s) for s in bseq)}", flush=True)
        if n_batches % args.checkpoint_every == 0:
            torch.save(emb, ckpt)

    torch.save(emb, ckpt)
    # final save (plain dict of tensors)
    torch.save(emb, args.out)
    dim = next(iter(emb.values())).numel() if emb else 0
    print(f"\n[done] wrote {args.out}", flush=True)
    print(f"       {len(emb):,} embeddings, dim={dim}, dtype={args.store_dtype}", flush=True)
    print(f"       elapsed {time.time()-t_run:.0f}s", flush=True)
    if dim != EXPECT_DIM:
        print(f"!! WARNING: dim {dim} != {EXPECT_DIM}", file=sys.stderr)


if __name__ == "__main__":
    main()
