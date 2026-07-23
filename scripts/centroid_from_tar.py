"""Iterate a local tar.gz of author UCE h5ads; for each .h5ad member, extract to
a temp file, compute per-cell-type centroids of the 1280-d UCE embedding (chunked,
low RAM via h5py), write <name>.centroids.npz, then delete the temp file.
Keeps peak disk = tarball + one member."""
import sys, os, json, re, tarfile
import numpy as np, h5py

TAR = sys.argv[1] if len(sys.argv) > 1 else '/workspace/other.tar.gz'
OUT = sys.argv[2] if len(sys.argv) > 2 else '/workspace/out'
TMP = sys.argv[3] if len(sys.argv) > 3 else '/workspace/tmp_member.h5ad'
os.makedirs(OUT, exist_ok=True)
CT_RE = re.compile(r'(cell.?type|celltype|cell_ontology|free_annotation|annotation|labels?)', re.I)

def read_cat(g):
    if isinstance(g, h5py.Group):
        cats = g['categories'][:]; codes = g['codes'][:]
        cats = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in cats])
        out = np.empty(codes.shape[0], dtype=object)
        valid = codes >= 0
        out[valid] = cats[codes[valid]]; out[~valid] = 'nan'
        return out.astype(str)
    arr = g[:]
    if arr.dtype.kind in ('S', 'O'):
        return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in arr])
    return arr.astype(str)

def pick_col(obs):
    keys = [k for k in obs.keys() if k not in ('__categories', '_index')]
    def ncat(k):
        g = obs[k]
        try:
            if isinstance(g, h5py.Group): return len(g['categories'])
            return len(np.unique(g[:min(len(g),200000)]))
        except Exception: return 0
    cand = [(k, ncat(k)) for k in keys]
    named = [(k, n) for k, n in cand if CT_RE.search(k) and 2 <= n <= 2000]
    if named:
        named.sort(key=lambda x: -x[1]); return named[0][0]
    generic = [(k, n) for k, n in cand if 2 <= n <= 500]
    generic.sort(key=lambda x: -x[1])
    return generic[0][0] if generic else None

def embed_dataset(f):
    if 'obsm' in f and 'X_uce' in f['obsm']: return f['obsm/X_uce']
    if 'obsm' in f:
        for k in f['obsm'].keys():
            d = f['obsm'][k]
            if isinstance(d, h5py.Dataset) and d.ndim == 2 and d.shape[1] == 1280: return d
    if 'X' in f and isinstance(f['X'], h5py.Dataset) and f['X'].ndim == 2 and f['X'].shape[1] == 1280:
        return f['X']
    return None

def process(path, name, summary):
    with h5py.File(path, 'r') as f:
        D = embed_dataset(f)
        if D is None:
            summary.append({'file': name, 'error': 'no 1280-d embedding',
                            'obsm': list(f.get('obsm', {}).keys())}); print('SKIP', name, 'no embed'); return
        N, dim = D.shape
        obs = f['obs']; col = pick_col(obs)
        if col is None:
            summary.append({'file': name, 'n_cells': int(N), 'error': 'no celltype col',
                            'obs_keys': list(obs.keys())}); print('SKIP', name, 'no col'); return
        labels = read_cat(obs[col]); uniq = np.unique(labels)
        lab_idx = {u: i for i, u in enumerate(uniq)}; K = len(uniq)
        sums = np.zeros((K, dim), np.float64); counts = np.zeros(K, np.int64)
        codes = np.array([lab_idx[l] for l in labels], np.int64)
        CH = 50000
        for s in range(0, N, CH):
            e = min(N, s + CH)
            block = np.asarray(D[s:e], np.float64)
            np.add.at(sums, codes[s:e], block); np.add.at(counts, codes[s:e], 1)
        cent = (sums / np.maximum(counts, 1)[:, None]).astype(np.float32)
        outp = os.path.join(OUT, name.replace('.h5ad', '') + '.centroids.npz')
        np.savez(outp, centroids=cent, labels=uniq.astype(str), counts=counts)
        summary.append({'file': name, 'n_cells': int(N), 'embed_dim': int(dim),
                        'celltype_col': col, 'n_types': int(K),
                        'types': {str(u): int(c) for u, c in zip(uniq, counts)}})
        print('OK', name, 'N=%d col=%s K=%d' % (N, col, K))

summary = []
with tarfile.open(TAR, 'r:gz') as tar:
    for m in tar:
        if not m.name.endswith('.h5ad'): continue
        name = os.path.basename(m.name)
        print('--> extracting', name, round(m.size/1e6, 1), 'MB')
        try:
            with open(TMP, 'wb') as w:
                src = tar.extractfile(m)
                while True:
                    chunk = src.read(1 << 24)
                    if not chunk: break
                    w.write(chunk)
            process(TMP, name, summary)
        except Exception as ex:
            summary.append({'file': name, 'error': repr(ex)}); print('ERR', name, repr(ex))
        finally:
            if os.path.exists(TMP): os.remove(TMP)
        json.dump(summary, open(os.path.join(OUT, 'ima_summary.json'), 'w'), indent=1)
print('DONE summary entries:', len(summary))
