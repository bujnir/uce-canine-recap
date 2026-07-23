"""Dog-intrinsic marker analysis: differential expression (Wilcoxon) between the
dog author cell types on the dog's OWN expression, top-5 markers per type, and a
heatmap of mean (and median) expression per cell type (z-scored per gene)."""
import scanpy as sc, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

dog = sc.read_h5ad('data/dog_eval_enscafg.h5ad')
dog.var_names = dog.var['gene_symbols'].astype(str).str.upper().values
dog = dog[:, dog.var_names != 'NAN'].copy(); dog.var_names_make_unique()
lab = pd.read_csv('out/dog_eval_labels.csv', index_col=0)
dog = dog[dog.obs_names.isin(lab.index)].copy()
dog.obs['l1'] = pd.Categorical(lab['celltype.l1'].reindex(dog.obs_names).values)
sc.pp.normalize_total(dog, target_sum=1e4); sc.pp.log1p(dog)
print('dog:', dog.n_obs, 'cells', dog.n_vars, 'genes; types:', list(dog.obs['l1'].cat.categories))

# order cell types in a lineage-sensible way
order = [t for t in ['CD4 T cell','CD8/NK cell','gd T cell','DN T cell','Cycling T cell',
                     'B cell','Plasma cell','Monocyte','DC','Neutrophil','Granulocyte',
                     'CD34+ Unclassified'] if t in set(dog.obs['l1'])]
dog.obs['l1'] = dog.obs['l1'].cat.reorder_categories(order)

# ---- DE: top 5 markers per type ----
sc.tl.rank_genes_groups(dog, 'l1', method='wilcoxon', n_genes=15)
top = {}
seen = set(); genes = []
for t in order:
    names = [g for g in dog.uns['rank_genes_groups']['names'][t]]
    picks = []
    for g in names:
        if g not in seen:
            picks.append(g); seen.add(g)
        if len(picks) == 5: break
    top[t] = picks; genes += picks
pd.DataFrame({t: top[t] for t in order}).to_csv('eval_out/dog_top5_markers.csv', index=False)
print('\nTop-5 dog markers per type:')
for t in order: print(f"  {t:20s}: {', '.join(top[t])}")

# ---- mean & median expression per type over selected marker genes ----
X = dog[:, genes].X
X = np.asarray(X.todense()) if hasattr(X, 'todense') else np.asarray(X)
df = pd.DataFrame(X, columns=genes); df['l1'] = dog.obs['l1'].values
mean_mat = df.groupby('l1', observed=True)[genes].mean().reindex(order)
med_mat = df.groupby('l1', observed=True)[genes].median().reindex(order)
mean_mat.to_csv('eval_out/dog_marker_mean_expr.csv')
med_mat.to_csv('eval_out/dog_marker_median_expr.csv')

# z-score per gene (column) across cell types, for the heatmap
def zscore(M):
    Z = (M - M.mean(0)) / (M.std(0) + 1e-9); return Z
Z = zscore(mean_mat)

fig, ax = plt.subplots(figsize=(max(12, len(genes) * 0.28), 6))
im = ax.imshow(Z.values, cmap='RdBu_r', vmin=-2, vmax=2, aspect='auto')
ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=9)
ax.set_xticks(range(len(genes))); ax.set_xticklabels(genes, rotation=90, fontsize=6.5)
# group separators by the type each block of 5 belongs to
b = 0
for t in order:
    n = len(top[t])
    ax.add_patch(plt.Rectangle((b - .5, order.index(t) - .5), n, 1, fill=False, ec='black', lw=1.4))
    b += n
ax.set_title('Dog-intrinsic markers: top-5 DE genes per author cell type\n(mean log-normalised expression, z-scored per gene)')
plt.colorbar(im, label='z-score of mean expression', shrink=0.6)
plt.tight_layout(); plt.savefig('eval_out/dog_marker_heatmap.png', dpi=150); plt.close()
print('\nsaved eval_out/dog_marker_heatmap.png + dog_top5_markers.csv + mean/median CSVs')
