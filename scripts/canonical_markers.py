"""Literature-curated canonical marker panel per canine leukocyte type.
Sources: Ammons et al. 2023 Front Immunol (10.3389/fimmu.2023.1162700) Table 3
(the dataset's own paper) + canonical pan-mammalian immune lineage markers.
Produces (a) a marker heatmap of dog expression per author type, and
(b) a marker-score annotation baseline (sc.tl.score_genes -> argmax)."""
import scanpy as sc, numpy as np, pandas as pd, json, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# canonical panel (human/dog gene symbols). * = pan-mammalian canonical; others from Ammons Table 3
PANEL = {
 'CD4 T cell':        ['CD4','IL7R','LEF1','CCR7','CD40LG','LTB'],
 'CD8/NK cell':       ['CD8A','GZMK','GZMB','GZMA','NCR3','KLRB1','KLRF1','CD96'],
 'gd T cell':         ['TRDC','ZNF683','IL17RB','TGFBR3','GATA3'],
 'DN T cell':         ['CD3E','CD3G','CTLA4','ID3','NMB'],
 'Cycling T cell':    ['MKI67','TOP2A','RRM2','TK1','STMN1'],
 'B cell':            ['MS4A1','CD79A','CD79B','PAX5','BANK1','TNFRSF13C','EBF1'],
 'Plasma cell':       ['JCHAIN','MZB1','DERL3','TXNDC5','TNFRSF13B'],
 'Monocyte':          ['LYZ','CSF1R','LRMDA','F13A1','S100A12','CD14'],
 'DC':                ['FLT3','FCER1A','CD1C','BATF3','IL3RA','ZNF366'],
 'Neutrophil':        ['S100A8','SERPINA1','CSF3R','IL1R2','SRGN'],
 'Granulocyte':       ['IL5RA','CA8','DACH1','PADI3','TGM2'],
 'CD34+ Unclassified':['CD34','KIT','ZNF521','CD109','DNTT','TFPI'],
}
order = list(PANEL.keys())

dog = sc.read_h5ad('data/dog_eval_enscafg.h5ad')
dog.var_names = dog.var['gene_symbols'].astype(str).str.upper().values
dog = dog[:, dog.var_names != 'NAN'].copy(); dog.var_names_make_unique()
lab = pd.read_csv('out/dog_eval_labels.csv', index_col=0)
dog = dog[dog.obs_names.isin(lab.index)].copy()
dog.obs['l1'] = pd.Categorical(lab['celltype.l1'].reindex(dog.obs_names).values, categories=order)
sc.pp.normalize_total(dog, target_sum=1e4); sc.pp.log1p(dog)
present = set(dog.var_names)

# report marker availability in dog (cross-species symbol coverage)
avail = {t: [g for g in gs if g in present] for t, gs in PANEL.items()}
missing = {t: [g for g in gs if g not in present] for t, gs in PANEL.items()}
print("marker availability in dog genome (symbol match):")
for t in order:
    print(f"  {t:20s}: {len(avail[t])}/{len(PANEL[t])} present; missing={missing[t]}")

# ---- (b) marker-score annotation ----
for t in order:
    if avail[t]:
        sc.tl.score_genes(dog, avail[t], score_name=f'sc_{t}')
S = np.vstack([dog.obs[f'sc_{t}'].values if f'sc_{t}' in dog.obs else np.full(dog.n_obs,-1e9) for t in order]).T
pred = np.array(order)[S.argmax(1)]
y = dog.obs['l1'].astype(str).values
acc = float((pred == y).mean())
labs = [c for c in order if c in set(y)]
mF = f1_score(y, pred, labels=labs, average='macro', zero_division=0)
rec = {c: round(float((pred[y == c] == c).mean()), 2) for c in order if (y == c).any()}
print(f"\n[Literature marker-score] all {len(y)} cells: acc={acc:.3f} macro-F1={mF:.3f}")
print("  per-type recall:", rec)
json.dump({'acc': round(acc,3),'macroF1': round(mF,3),'n': int(len(y)),'recall': rec,
           'missing_markers': missing},
          open('eval_out/marker_score_results.json','w'), indent=1)
pd.DataFrame({'l1': y, 'marker_pred': pred}, index=dog.obs_names).to_csv('eval_out/marker_score_pred.csv')

# ---- (a) marker heatmap: mean dog expression per type, z-scored per gene ----
genes = []
for t in order:
    for g in avail[t]:
        if g not in genes: genes.append(g)
X = dog[:, genes].X; X = np.asarray(X.todense()) if hasattr(X,'todense') else np.asarray(X)
df = pd.DataFrame(X, columns=genes); df['l1'] = y
M = df.groupby('l1', observed=True)[genes].mean().reindex(order)
M.to_csv('eval_out/canonical_marker_mean_expr.csv')
Z = ((M - M.mean(0)) / (M.std(0) + 1e-9))
fig, ax = plt.subplots(figsize=(max(13, len(genes)*0.3), 6.2))
im = ax.imshow(Z.values, cmap='RdBu_r', vmin=-2, vmax=2, aspect='auto')
ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=9)
ax.set_xticks(range(len(genes))); ax.set_xticklabels(genes, rotation=90, fontsize=7)
# outline the block of markers belonging to each type on its own row
col = 0
for t in order:
    n = len(avail[t])
    if n: ax.add_patch(plt.Rectangle((col-.5, order.index(t)-.5), n, 1, fill=False, ec='k', lw=1.4))
    col += n
ax.set_title('Literature canonical markers (Ammons 2023 + canonical) in dog cells\nmean log-normalised expression, z-scored per gene')
plt.colorbar(im, label='z-score', shrink=0.6); plt.tight_layout()
plt.savefig('eval_out/canonical_marker_heatmap.png', dpi=150); plt.close()
print('\nsaved eval_out/canonical_marker_heatmap.png + marker_score_results.json + CSVs')
