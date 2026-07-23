"""
Build UCE dog new-species files (replicates data_proc/Create New Species Files.ipynb).
ENSCAFG keying throughout (matches dog_esm2_embeddings.pt keys + dog h5ad var_names).

Outputs:
  dog_to_chrom_pos.csv   (gene_symbol=ENSCAFG upper, chromosome, start, species=dog)  [no figshare needed]
  dog_offsets.pkl        ({"dog": 4})                                                  [no figshare needed]
  dog_pe_tokens.torch    (all_tokens[0:4] special + dog PE + CHROM_TENSORS)            [NEEDS all_tokens.torch]
  prints CHROM_TOKEN_OFFSET
"""
import argparse, pickle, pandas as pd, numpy as np, torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gene-table', default='data/dog_gene_table.csv')
    ap.add_argument('--pe', default='dog_esm2_embeddings.pt')
    ap.add_argument('--all-tokens', default=None, help='UCE all_tokens.torch (for special rows); if omitted, uses zeros placeholder (TEST ONLY)')
    ap.add_argument('--taxonomy', type=int, default=9615)
    ap.add_argument('--outdir', default='newspecies')
    ap.add_argument('--species', default='dog')
    args = ap.parse_args()

    gt = pd.read_csv(args.gene_table)
    # ENSCAFG keying: gene_symbol column = uppercase ENSCAFG gene_id; fill missing chrom/start
    df = pd.DataFrame({
        'gene_symbol': gt['gene_id'].str.upper(),
        'chromosome':  gt['chromosome'].fillna('U').astype(str),
        'start':       gt['start'].fillna(0).astype(np.int64),
    })
    df['species'] = args.species
    df = df.sort_values(['chromosome','start'])
    csv_path = f"{args.outdir}/{args.species}_to_chrom_pos.csv"
    df.to_csv(csv_path, index=False)
    N_UNIQ_CHROM = df['chromosome'].nunique()
    print(f"wrote {csv_path}: {len(df)} genes, {N_UNIQ_CHROM} unique chromosomes")

    # protein embeddings (dict {ENSCAFG: tensor[5120]})
    PE = torch.load(args.pe, map_location='cpu')
    pe_stacked = torch.stack([v.float() for v in PE.values()])   # (n_genes, 5120) in PE key order
    n_genes, dim = pe_stacked.shape
    assert dim == 5120, dim

    # special tokens (first 4 rows of UCE all_tokens.torch)
    if args.all_tokens:
        special = torch.load(args.all_tokens, map_location='cpu')[0:4].float()
    else:
        print("!! no --all-tokens: using ZERO placeholder special rows (TEST ONLY, not for real inference)")
        special = torch.zeros(4, 5120)
    offset = special.shape[0]  # 4

    all_pe = torch.vstack((special, pe_stacked))
    CHROM_TOKEN_OFFSET = all_pe.shape[0]           # 4 + n_genes
    print("CHROM_TOKEN_OFFSET:", CHROM_TOKEN_OFFSET)

    torch.manual_seed(args.taxonomy)
    CHROM_TENSORS = torch.normal(mean=0, std=1, size=(N_UNIQ_CHROM, 5120))
    all_pe = torch.vstack((all_pe, CHROM_TENSORS))
    all_pe.requires_grad = False
    tok_path = f"{args.outdir}/{args.species}_pe_tokens.torch"
    torch.save(all_pe, tok_path)

    with open(f"{args.outdir}/{args.species}_offsets.pkl", 'wb') as f:
        pickle.dump({args.species: offset}, f)

    print(f"wrote {tok_path}: shape {tuple(all_pe.shape)} = 4 special + {n_genes} PE + {N_UNIQ_CHROM} chrom")
    print(f"wrote {args.outdir}/{args.species}_offsets.pkl: {{'{args.species}': {offset}}}")
    print(f"\n>>> pass at inference:  --CHROM_TOKEN_OFFSET {CHROM_TOKEN_OFFSET}")

if __name__ == '__main__':
    main()
