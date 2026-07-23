"""Build UCE-ready dog h5ad: raw counts, var_names = ENSCAFG gene_ids, obs labels.
Pod version downloads RAW.tar (GEO) + meta (UCSC); here we use local copies."""
import argparse, os, glob, tarfile, re
import scanpy as sc, anndata as ad, pandas as pd, numpy as np, scipy.sparse as sp

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--raw-tar', default='data/GSE225599_RAW.tar')
    ap.add_argument('--meta', default=None)
    ap.add_argument('--out', default='data/dog_eval_enscafg.h5ad')
    ap.add_argument('--workdir', default='/tmp/dogmtx')
    a=ap.parse_args()
    os.makedirs(a.workdir, exist_ok=True)
    with tarfile.open(a.raw_tar) as t: t.extractall(a.workdir)
    samples=sorted({os.path.basename(f)[:-len('_barcodes.tsv.gz')]
                    for f in glob.glob(a.workdir+'/GSM*_barcodes.tsv.gz')})
    adatas=[]
    for s in samples:
        d=os.path.join(a.workdir,s); os.makedirs(d,exist_ok=True)
        for suf in ['barcodes.tsv.gz','features.tsv.gz','matrix.mtx.gz']:
            src=glob.glob(f"{a.workdir}/{s}_{suf}")[0]
            os.replace(src, f"{d}/{suf}")
        A=sc.read_10x_mtx(d, var_names='gene_ids', make_unique=True)  # ENSCAFG
        cond='healthy' if re.search(r'_H_',s) else 'osteosarcoma'
        A.obs['sample']=s; A.obs['condition']=cond
        A.obs_names=[f"{s}_{bc}" for bc in A.obs_names]
        adatas.append(A)
    A=ad.concat(adatas, join='outer', merge='same'); A.X=sp.csr_matrix(A.X)
    if a.meta and os.path.exists(a.meta):
        m=pd.read_csv(a.meta, sep='\t', low_memory=False)
        m['barcode']=m['Cell'].str.rsplit('_',n=1).str[0]; m['key']=m['name'].astype(str)+'|'+m['barcode']
        lab=m.set_index('key'); samp=A.obs['sample'].astype(str); name=samp.str.split('_',n=1).str[1]
        bc=[on[len(s)+1:] for on,s in zip(A.obs_names,samp)]; A.obs['key']=name.values+'|'+np.array(bc)
        for c in ['celltype.l1','celltype.l2','celltype.l3']: A.obs[c]=lab[c].reindex(A.obs['key']).values
        E=A[A.obs['celltype.l1'].notna()].copy(); del E.obs['key']
    else:
        E=A  # all cells, no labels (labels joined later in eval by cell id)
    E.write_h5ad(a.out)
    print(f"wrote {a.out}: {E.shape}")
    print("var_names sample:", list(E.var_names[:3]), "| all ENSCAFG:", all(str(v).startswith('ENSCAFG') for v in E.var_names[:100]))
    print("int counts:", float(abs(E.X.data-E.X.data.round()).max()))
    if 'celltype.l1' in E.obs: print("celltype.l1:", E.obs['celltype.l1'].value_counts().to_dict())

if __name__=='__main__': main()
