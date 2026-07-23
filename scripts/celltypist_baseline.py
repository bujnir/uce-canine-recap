"""Conventional baseline: CellTypist pretrained human immune model on dog cells,
mapped to human gene symbols (cross-species the standard way). Score vs author labels."""
import scanpy as sc, numpy as np, pandas as pd, celltypist, json
from celltypist import models
from sklearn.metrics import f1_score

# ---- load dog counts, key by human-orthologous gene symbol ----
ad = sc.read_h5ad('data/dog_eval_enscafg.h5ad')
sym = ad.var['gene_symbols'].astype(str).str.upper().values
ad.var['sym'] = sym
ad = ad[:, sym != 'NAN']
ad.var_names = ad.var['sym'].values
ad.var_names_make_unique()
# restrict to cells with author labels
lab = pd.read_csv('out/dog_eval_labels.csv', index_col=0)
ad = ad[ad.obs_names.isin(lab.index)].copy()
ad.obs['l1'] = lab['celltype.l1'].reindex(ad.obs_names).values
print('dog cells:', ad.n_obs, 'genes:', ad.n_vars)

# ---- normalise to CellTypist's expected log1p CP10k ----
sc.pp.normalize_total(ad, target_sum=1e4)
sc.pp.log1p(ad)

# ---- coarse map from CellTypist labels -> dog coarse space ----
def coarse(ct):
    c = str(ct).lower()
    if 'plasma' in c: return 'Plasma cell'
    if 'gamma-delta' in c or 'gdt' in c or 'γδ' in c: return 'gd T cell'
    if 'double-negative' in c or 'double negative' in c: return 'DN T cell'
    if 'nk' in c or 'cytotox' in c or 'mait' in c: return 'CD8/NK cell'
    if 'cd8' in c or 'tem/' in c: return 'CD8/NK cell'
    if 'cd4' in c or 'treg' in c or 'regulatory' in c or 'helper' in c or 'follicular' in c or 'tfh' in c or 'th1' in c or 'th2' in c or 'th17' in c: return 'CD4 T cell'
    if c.strip() in ('t cells','abt(entry)','cycling t cells') or 'thymocyte' in c: return 'T cell (unsplit)'
    if 'monocyte' in c or 'macrophage' in c or 'mono-mac' in c or 'kupffer' in c: return 'Monocyte'
    if 'dc' in c or 'dendritic' in c: return 'DC'
    if 'neutrophil' in c or 'granulocyte' in c: return 'Neutrophil'
    if 'b cell' in c or 'b-cell' in c or 'germinal' in c or 'memory b' in c or 'naive b' in c or 'plasmablast' in c or 'pro-b' in c or 'pre-b' in c: return 'B cell'
    if 'hsc' in c or 'mpp' in c or 'progenitor' in c or 'cd34' in c or 'stem' in c or 'gmp' in c or 'clp' in c or 'cmp' in c: return 'CD34+ Unclassified'
    if 'mast' in c: return 'Mast'
    if 'ilc' in c: return 'ILC'
    return 'other'

results = {}
for model_name in ['Immune_All_Low.pkl', 'Immune_All_High.pkl']:
    pred = celltypist.annotate(ad, model=model_name, majority_voting=False)
    raw = pred.predicted_labels['predicted_labels'].values
    cp = np.array([coarse(x) for x in raw])
    y = ad.obs['l1'].values
    predset = sorted(set([c for c in cp if c not in ('other','T cell (unsplit)','Mast','ILC')]))
    ev = np.array([t in set(predset) for t in y])
    acc = float((cp[ev] == y[ev]).mean())
    labs = [c for c in predset if c in set(y)]
    mF = f1_score(y[ev], cp[ev], labels=labs, average='macro', zero_division=0)
    rec = {c: round(float((cp[y == c] == c).mean()), 2) for c in pd.unique(y) if c in set(predset)}
    tag = model_name.replace('.pkl', '')
    results[tag] = {'acc': round(acc, 3), 'macroF1': round(mF, 3), 'evaluable': int(ev.sum()), 'recall': rec}
    print(f"\n[CellTypist {tag}] evaluable={int(ev.sum())} acc={acc:.3f} macro-F1={mF:.3f}")
    print("  per-type recall:", rec)
    # save per-cell predictions
    pd.DataFrame({'l1': y, 'celltypist_raw': raw, 'celltypist_coarse': cp},
                 index=ad.obs_names).to_csv(f'eval_out/celltypist_{tag}_pred.csv')

json.dump(results, open('eval_out/celltypist_results.json', 'w'), indent=1)
print("\nsaved eval_out/celltypist_results.json")
