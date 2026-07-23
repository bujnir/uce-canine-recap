import scanpy as sc, numpy as np, sys
inp, prefix = sys.argv[1], sys.argv[2]
a = sc.read_h5ad(inp)
X = np.asarray(a.obsm['X_uce']).astype('float16')
np.save(prefix + '_X_uce.npy', X)
a.obs.to_csv(prefix + '_obs.csv')
print(f"saved {prefix}_X_uce.npy {X.shape} + {prefix}_obs.csv ({a.n_obs} rows)")
