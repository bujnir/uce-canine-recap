"""Zero-shot annotation of dog cells against the AUTHORS' own UCE-embedded
Tabula Sapiens atlas (member1 of Zenodo other_files.tar.gz), with authoritative
TS cell-type labels recovered by cell-ID join to the CZ CELLxGENE TS obs.

Two methods:
  (A) kNN over author reference CELLS  -> the "drop my cells on their map, read
      the label of the nearest neighbours" idea (proximity on the UMAP).
  (B) nearest-centroid over reference cell-type centroids -> the paper's method.
Both use per-species mean-centering to remove the cross-species offset.
"""
import numpy as np, pandas as pd, os, h5py, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors
os.makedirs('eval_out', exist_ok=True)

# ---------- load author reference (member1) + recover TS labels ----------
mf = h5py.File('/tmp/member1.h5ad', 'r')
Xr_all = np.asarray(mf['obsm']['X_uce'][:], dtype='float32')          # 100000 x 1280
rid = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in mf['obs']['_index'][:]])
tsobs = pd.read_parquet('/tmp/ts_full_obs.parquet')
lut = dict(zip(tsobs.cell_id.values.tolist(), tsobs.cell_type.values.tolist()))
def reorder(s):
    p = s.split('_'); return '_'.join([p[-1]] + p[:-1])   # move barcode token to front -> CZ format
rlab_fine = np.array([lut.get(reorder(x)) for x in rid], dtype=object)
keep = pd.notna(rlab_fine)
Xr = Xr_all[keep]; rlab_fine = rlab_fine[keep].astype(str)
print(f"author reference: {Xr.shape[0]} labelled cells, {len(set(rlab_fine))} fine TS types")

# ---------- coarse mapping (same as prior evals, for comparability) ----------
def coarse(ct):
    c = ct.lower()
    if 'plasma cell' in c or 'plasmablast' in c: return 'Plasma cell'
    if 'gamma-delta' in c: return 'gd T cell'
    if 'double negative' in c or 'double-negative' in c: return 'DN T cell'
    if 'natural killer' in c or 'nk t' in c or ' nk' in c: return 'CD8/NK cell'
    if 'cd8' in c or 'mucosal invariant' in c: return 'CD8/NK cell'
    if 'cd4' in c or 'regulatory t' in c or 'follicular helper' in c or 't-helper' in c or 'helper t' in c: return 'CD4 T cell'
    if 'monocyte' in c or 'macrophage' in c: return 'Monocyte'
    if 'dendritic' in c: return 'DC'
    if 'neutrophil' in c: return 'Neutrophil'
    if 'b cell' in c or 'memory b' in c or 'naive b' in c or 'immature b' in c or 'plasmablast' in c: return 'B cell'
    if 'hematopoietic' in c or 'progenitor' in c or 'precursor' in c or 'cd34' in c: return 'CD34+ Unclassified'
    return 'other'
rc = np.array([coarse(x) for x in rlab_fine])

# ---------- load dog ----------
Xd = np.load('uce_out/dog_X_uce.npy').astype('float32')
od = pd.read_csv('uce_out/dog_obs.csv', index_col=0)
lab = pd.read_csv('out/dog_eval_labels.csv', index_col=0)
mask = od.index.isin(lab.index); Xd = Xd[mask]; od = od[mask].copy()
od['l1'] = lab['celltype.l1'].reindex(od.index).values
od['condition'] = lab['condition'].reindex(od.index).values
y = od['l1'].values
print(f"dog: {Xd.shape[0]} cells with author labels")

# ---------- species-center + L2-normalise (cosine geometry) ----------
offset = float(np.linalg.norm(Xd.mean(0) - Xr.mean(0)))
def cen_norm(X, mu):
    Z = X - mu
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    return Z
Zr = cen_norm(Xr, Xr.mean(0))
Zd = cen_norm(Xd, Xd.mean(0))
print(f"species offset removed: {offset:.2f}")

predset_coarse = sorted(set(rc[rc != 'other']))

# ---------- (A) kNN over reference cells (proximity on the map) ----------
K = 15
ref_kept = rc != 'other'
nn = NearestNeighbors(n_neighbors=K, metric='cosine').fit(Zr[ref_kept])
dist, ind = nn.kneighbors(Zd)
neigh_lab = rc[ref_kept][ind]                       # (n_dog, K)
def majority(row):
    u, c = np.unique(row, return_counts=True); j = c.argmax(); return u[j], c[j] / len(row)
knn_pred = np.empty(len(Zd), dtype=object); knn_conf = np.empty(len(Zd))
for i in range(len(Zd)):
    knn_pred[i], knn_conf[i] = majority(neigh_lab[i])
od['knn_pred'] = knn_pred; od['knn_conf'] = knn_conf

# ---------- (B) nearest-centroid (paper method) ----------
cnames = predset_coarse
cent = np.vstack([Zr[ref_kept][rc[ref_kept] == c].mean(0) for c in cnames])
cent = cent / (np.linalg.norm(cent, axis=1, keepdims=True) + 1e-8)
sim = Zd @ cent.T
nc_pred = np.array(cnames)[sim.argmax(1)]
od['nc_pred'] = nc_pred

# ---------- metrics ----------
def report(pred, name):
    ev = np.array([t in set(cnames) for t in y])
    acc = float((pred[ev] == y[ev]).mean())
    labs = [c for c in cnames if c in set(y)]
    mF = f1_score(y[ev], pred[ev], labels=labs, average='macro', zero_division=0)
    rec = {c: round(float((pred[y == c] == c).mean()), 2) for c in pd.unique(y) if c in set(cnames)}
    print(f"\n[{name}] evaluable={int(ev.sum())}  accuracy={acc:.3f}  macro-F1={mF:.3f}")
    print("  per-type recall:", rec)
    return acc, mF, rec
accA, mFA, recA = report(knn_pred, 'kNN over author cells (k=15)')
accB, mFB, recB = report(nc_pred, 'nearest-centroid (paper method)')

# ---------- confusion matrix (kNN), ordered for clean diagonal ----------
def ordered_confusion(pred, fname, title):
    rows = [r for r in ['CD4 T cell','CD8/NK cell','gd T cell','DN T cell','B cell','Plasma cell',
                        'Monocyte','DC','Neutrophil','CD34+ Unclassified'] if r in set(y)]
    cols = cnames
    P = np.zeros((len(rows), len(cols)))
    for i, r in enumerate(rows):
        m = y == r
        for j, c in enumerate(cols):
            P[i, j] = (pred[m] == c).mean() if m.sum() else 0
    # order cols to put best match per row on diagonal
    col_order, used = [], set()
    for i in range(len(rows)):
        order = np.argsort(-P[i])
        for j in order:
            if cols[j] not in used:
                col_order.append(j); used.add(cols[j]); break
    for j in range(len(cols)):
        if cols[j] not in used: col_order.append(j); used.add(cols[j])
    cols2 = [cols[j] for j in col_order]; P = P[:, col_order]
    pd.DataFrame(P, index=rows, columns=cols2).round(3).to_csv(f'eval_out/{fname}.csv')
    fig, ax = plt.subplots(figsize=(8.5, 6))
    im = ax.imshow(P, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(cols2))); ax.set_xticklabels(cols2, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=8)
    ax.set_xlabel('Predicted (nearest author cells / IMA)'); ax.set_ylabel('Dog author label')
    ax.set_title(title)
    for i in range(len(rows)):
        for j in range(len(cols2)):
            if P[i, j] > 0.12:
                ax.text(j, i, f'{P[i,j]:.2f}', ha='center', va='center', fontsize=7,
                        color='white' if P[i, j] > 0.5 else 'black')
    plt.colorbar(im, label='fraction of dog cells'); plt.tight_layout()
    plt.savefig(f'eval_out/{fname}.png', dpi=140); plt.close()
ordered_confusion(knn_pred, 'confusion_ima_knn',
                  f'Dog cells dropped onto authors\' TS UCE map (kNN, k=15)\nacc={accA:.2f}  macro-F1={mFA:.2f}')

# ---------- centroid matching: which fine TS type does each dog cluster land on ----------
fine_keep = pd.Series(rlab_fine).groupby(rlab_fine).transform('size').values >= 30
fn = sorted(set(rlab_fine[fine_keep]))
fcent = np.vstack([Zr[rlab_fine == f].mean(0) for f in fn])
fcent = fcent / (np.linalg.norm(fcent, axis=1, keepdims=True) + 1e-8)
rowsd = []
for dc in pd.unique(y):
    v = Zd[y == dc].mean(0); v = v / (np.linalg.norm(v) + 1e-8)
    o = (fcent @ v).argsort()[::-1]
    rowsd.append({'dog_celltype': dc, 'n_dog': int((y == dc).sum()),
                  'nearest_TS': fn[o[0]], '2nd': fn[o[1]], '3rd': fn[o[2]]})
pd.DataFrame(rowsd).to_csv('eval_out/centroid_matching_ima.csv', index=False)
print("\nDog cluster -> nearest AUTHOR (Tabula Sapiens) cell type:")
print(pd.DataFrame(rowsd).to_string(index=False))

# ---------- healthy vs OS (kNN predictions) ----------
prop = od.groupby('condition')['knn_pred'].value_counts(normalize=True).unstack().fillna(0)
prop.T.round(3).to_csv('eval_out/healthy_vs_os_ima.csv')

od[['sample','condition','l1','knn_pred','knn_conf','nc_pred']].to_csv('eval_out/dog_predictions_ima.csv')

summary = {'ref_cells': int(Xr.shape[0]), 'ref_fine_types': int(len(set(rlab_fine))),
           'species_offset': round(offset, 3),
           'knn_acc': round(accA, 3), 'knn_macroF1': round(mFA, 3),
           'centroid_acc': round(accB, 3), 'centroid_macroF1': round(mFB, 3)}
import json; json.dump(summary, open('eval_out/ima_eval_summary.json', 'w'), indent=1)
print("\nSUMMARY", summary)
print("saved eval_out/: confusion_ima_knn.png, centroid_matching_ima.csv, dog_predictions_ima.csv")
