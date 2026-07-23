#!/usr/bin/env python3
"""
Step 01 - Parse the Ensembl dog peptide FASTA into:
  (a) dog_gene_table.csv  : gene_id, gene_symbol, chromosome, start, prot_id, length
                            one row per gene (longest isoform kept)
  (b) dog_longest.pep.fa  : one protein sequence per gene -> input to ESM2

Why longest-isoform: UCE/SATURN average protein embeddings per gene. Averaging
is fine, but for a first pass the longest isoform is cheaper (1 seq/gene instead
of ~2.5) and is the standard canonical proxy. Set --all-isoforms to embed every
isoform and average later.

Usage:
  python 01_parse_proteome.py --fasta data/Canis_lupus_familiaris.CanFam3.1.pep.all.fa.gz
"""
import argparse
import gzip
import re
import csv
from pathlib import Path

HDR = re.compile(
    r'^>(?P<prot>\S+)\s+.*?'
    r'chromosome:(?P<asm>[^:]+):(?P<chrom>[^:]+):(?P<start>\d+):(?P<end>\d+):(?P<strand>-?1)'
    r'.*?gene:(?P<gene>\S+)'
)
SYMBOL = re.compile(r'gene_symbol:(\S+)')
BIOTYPE = re.compile(r'gene_biotype:(\S+)')


def parse(fasta_path):
    """Yield dict per protein record."""
    opener = gzip.open if str(fasta_path).endswith('.gz') else open
    rec = None
    with opener(fasta_path, 'rt') as fh:
        for line in fh:
            if line.startswith('>'):
                if rec:
                    yield rec
                m = HDR.match(line)
                if not m:
                    # scaffold-level (non-chromosome) entries fall here; keep but flag
                    prot = line[1:].split()[0]
                    g = re.search(r'gene:(\S+)', line)
                    rec = {
                        'prot_id': prot,
                        'gene_id': g.group(1).split('.')[0] if g else None,
                        'chromosome': None, 'start': None,
                        'gene_symbol': (SYMBOL.search(line).group(1)
                                        if SYMBOL.search(line) else None),
                        'biotype': (BIOTYPE.search(line).group(1)
                                    if BIOTYPE.search(line) else None),
                        'seq': [],
                    }
                    continue
                d = m.groupdict()
                rec = {
                    'prot_id': d['prot'],
                    'gene_id': d['gene'].split('.')[0],   # strip version
                    'chromosome': d['chrom'],
                    'start': int(d['start']),
                    'gene_symbol': (SYMBOL.search(line).group(1)
                                    if SYMBOL.search(line) else None),
                    'biotype': (BIOTYPE.search(line).group(1)
                                if BIOTYPE.search(line) else None),
                    'seq': [],
                }
            elif rec is not None:
                rec['seq'].append(line.strip())
    if rec:
        yield rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fasta', required=True)
    ap.add_argument('--outdir', default='data')
    ap.add_argument('--all-isoforms', action='store_true',
                    help='keep every isoform instead of longest-per-gene')
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    best = {}     # gene_id -> record (longest)
    n_records = 0
    for rec in parse(args.fasta):
        rec['seq'] = ''.join(rec['seq'])
        rec['length'] = len(rec['seq'])
        n_records += 1
        if rec['gene_id'] is None:
            continue
        if args.all_isoforms:
            best[rec['prot_id']] = rec
        else:
            cur = best.get(rec['gene_id'])
            if cur is None or rec['length'] > cur['length']:
                best[rec['gene_id']] = rec

    genes = list(best.values())

    # ---- write gene table
    tbl = outdir / 'dog_gene_table.csv'
    with open(tbl, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['gene_id', 'gene_symbol', 'chromosome', 'start',
                    'prot_id', 'length', 'biotype'])
        for r in sorted(genes, key=lambda x: (str(x['chromosome']), x['start'] or 0)):
            w.writerow([r['gene_id'], r['gene_symbol'], r['chromosome'],
                        r['start'], r['prot_id'], r['length'], r['biotype']])

    # ---- write FASTA for ESM2 (id = gene_id, so downstream keying is trivial)
    fa = outdir / 'dog_longest.pep.fa'
    with open(fa, 'w') as fh:
        for r in genes:
            fh.write(f">{r['gene_id']}\n")
            for i in range(0, len(r['seq']), 60):
                fh.write(r['seq'][i:i + 60] + '\n')

    # ---- diagnostics: this is the number that decides the project
    n_genes = len(genes)
    n_sym = sum(1 for r in genes if r['gene_symbol'])
    n_chrom = sum(1 for r in genes if r['chromosome'] and
                  not str(r['chromosome']).startswith('AAEX'))
    n_long = sum(1 for r in genes if r['length'] > 1022)

    print(f"protein records parsed : {n_records:,}")
    print(f"unique genes           : {n_genes:,}")
    print(f"  with gene_symbol     : {n_sym:,}  ({100*n_sym/n_genes:.1f}%)")
    print(f"  WITHOUT gene_symbol  : {n_genes-n_sym:,}  <-- ENSCAFG-only, the mapping problem")
    print(f"  on named chromosome  : {n_chrom:,}")
    print(f"  seq >1022aa (ESM2 truncates): {n_long:,}")
    print(f"\nwrote {tbl}")
    print(f"wrote {fa}")


if __name__ == '__main__':
    main()
