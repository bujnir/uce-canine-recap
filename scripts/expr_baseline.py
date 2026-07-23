"""Controlled baseline: SAME authors' Tabula Sapiens reference (member1, 32k labelled)
but conventional EXPRESSION methods in shared human-orthologous-symbol space:
  (S) SingleR-style: Spearman correlation of each dog cell to per-type mean profiles.
  (L) Expression logistic regression trained on the reference.
  (K) Expression kNN in shared-gene PCA space.
Isolates the value of the UCE representation vs raw ortholog expression on the same ref.
"""
import scanpy as sc, numpy as np, pandas as pd, h5py, json
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score

def coarse(ct):
    c = str(ct).lower()
    if 'plasma cell' in c or 'plasmablast' in c: return 'Plasma cell'
    if 'gamma-delta' in c: return 'gd T cell'
    if 'double negative' in c or 'double-negative' in c: return 'DN T cell'
    if 'natural killer' in c or 'nk t' in c or ' nk' in c: return 'CD8/NK cell'
    if 'cd8' in c or 'mucosal invariant' in c: return 'CD8/NK cell'
    if 'cd4' in c or 'regulatory t' in c or 'follicular helper' in c or 't-helper' in c or 'helper t' in c: return 'CD4 T cell'
    if 'monocyte' in c or 'macrophage' in c: return 'Monocyte'
    if 'dendritic' in c: return 'DC'
    if 'neutrophil' in c: return 'Neutrophil'
    if 'b cell' in c or 'memory b' in c or 'naive b' in c or 'immature b' in c: return 'B cell'
    if 'hematopoietic' in c or 'progenitor' in c or 'precursor' in c or 'cd34' in c: return 'CD34+ Unclassified'
    return 'other'

# ---- reference: authors' TS (member1) + recovered labels ----
ref = sc.read_h5ad('/tmp/member1.h5ad')
ref.var_names = ref.var['gene_symbol'].astype(str).str.upper().values
ref.var_names_make_unique()
rid = ref.obs_names.values
tsobs = pd.read_parquet('/tmp/ts_full_obs.parquet'); lut = dict(zip(tsobs.cell_id, tsobs.cell_type))
def reorder(s): p = s.split('_'); return '_'.join([p[-1]] + p[:-1])
fine = np.array([lut.get(reorder(x)) for x in rid], dtype=object)
keep = pd.notna(fine)
ref = ref[keep].copy(); rc = np.array([coarse(x) for x in fine[keep]])
ref = ref[rc != 'other'].copy(); rc = rc[rc != 'other']
sc.pp.normalize_total(ref, target_sum=1e4); sc.pp.log1p(ref)
print('reference:', ref.n_obs, 'cells,', ref.n_vars, 'genes')

# ---- query: dog counts in symbol space ----
dog = sc.read_h5ad('data/dog_eval_enscafg.h5ad')
dog.var_names = dog.var['gene_symbols'].astype(str).str.upper().values
dog = dog[:, dog.var_names != 'NAN'].copy(); dog.var_names_make_unique()
lab = pd.read_csv('out/dog_eval_labels.csv', index_col=0)
dog = dog[dog.obs_names.isin(lab.index)].copy()
dog.obs['l1'] = lab['celltype.l1'].reindex(dog.obs_names).values
sc.pp.normalize_total(dog, target_sum=1e4); sc.pp.log1p(dog)
print('dog:', dog.n_obs, 'cells,', dog.n_vars, 'genes')

# ---- shared genes; pick top HVGs from reference ----
shared = sorted(set(ref.var_names) & set(dog.var_names))
print('shared genes:', len(shared))
refS = ref[:, shared].copy()
sc.pp.highly_variable_genes(refS, n_top_genes=2000)
hvg = list(refS.var_names[refS.var['highly_variable'].values])
Xr = (np.asarray(refS[:, hvg].X.todense()) if hasattr(refS.X, 'todense') else np.asarray(refS[:, hvg].X)).astype(np.float32)
Xd = (np.asarray(dog[:, hvg].X.todense()) if hasattr(dog.X, 'todense') else np.asarray(dog[:, hvg].X)).astype(np.float32)
y = dog.obs['l1'].values
classes = sorted(set(rc))
predset = classes

def score(pred, name, res):
    ev = np.array([t in set(predset) for t in y])
    acc = float((pred[ev] == y[ev]).mean())
    labs = [c for c in predset if c in set(y)]
    mF = f1_score(y[ev], pred[ev], labels=labs, average='macro', zero_division=0)
    rec = {c: round(float((pred[y == c] == c).mean()), 2) for c in pd.unique(y) if c in set(predset)}
    res[name] = {'acc': round(acc, 3), 'macroF1': round(mF, 3), 'evaluable': int(ev.sum()), 'recall': rec}
    print(f"\n[{name}] evaluable={int(ev.sum())} acc={acc:.3f} macro-F1={mF:.3f}\n  recall: {rec}")
    json.dump(res, open('eval_out/expr_baseline_results.json', 'w'), indent=1)  # progressive save

res = {}

# ---- (S) SingleR-style: Spearman corr to per-type mean profiles ----
prof = np.vstack([Xr[rc == c].mean(0) for c in classes])            # types x hvg
prof_r = rankdata(prof, axis=1)                                      # rank across genes (vectorised)
Xd_r = rankdata(Xd, axis=1)
pr = (prof_r - prof_r.mean(1, keepdims=True)); pr /= (np.linalg.norm(pr, axis=1, keepdims=True) + 1e-9)
dr = (Xd_r - Xd_r.mean(1, keepdims=True)); dr /= (np.linalg.norm(dr, axis=1, keepdims=True) + 1e-9)
corr = dr @ pr.T
singler = np.array(classes)[corr.argmax(1)]
score(singler, 'SingleR-style (Spearman, same TS ref)', res)

# ---- (L) expression logistic regression ----
ss = StandardScaler().fit(Xr)
clf = LogisticRegression(max_iter=2000, C=0.5, n_jobs=-1).fit(ss.transform(Xr), rc)
lrpred = clf.predict(ss.transform(Xd))
score(lrpred, 'Expression logistic-regression (same TS ref)', res)

# ---- (K) expression kNN in shared PCA space (memory-safe: fit on ref only) ----
sca = StandardScaler().fit(Xr)
pca = PCA(n_components=50, random_state=0).fit(sca.transform(Xr).astype(np.float32))
Zr = pca.transform(sca.transform(Xr).astype(np.float32)).astype(np.float32)
Zd = pca.transform(sca.transform(Xd).astype(np.float32)).astype(np.float32)
nn = NearestNeighbors(n_neighbors=15, metric='euclidean').fit(Zr)
_, ind = nn.kneighbors(Zd)
knn = np.array([pd.Series(rc[ind[i]]).value_counts().idxmax() for i in range(len(Zd))])
score(knn, 'Expression kNN k=15 (same TS ref)', res)

json.dump(res, open('eval_out/expr_baseline_results.json', 'w'), indent=1)
pd.DataFrame({'l1': y, 'singleR': singler, 'expr_LR': lrpred, 'expr_kNN': knn},
             index=dog.obs_names).to_csv('eval_out/expr_baseline_pred.csv')
print("\nsaved eval_out/expr_baseline_results.json")
