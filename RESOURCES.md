# Resources & data

Everything needed to reproduce the project. Keying throughout is Ensembl gene IDs (ENSCAFG).

| Step / purpose | Resource / data | Link |
|---|---|---|
| Dog scRNA-seq data | GEO **GSE225599** (Ammons 2023) | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE225599 |
| Cell-type labels & markers | Ammons et al. 2023, *Front. Immunol.* | https://doi.org/10.3389/fimmu.2023.1162700 |
| Dog proteome | Ensembl release-104, CanFam3.1 (peptide FASTA) | https://ftp.ensembl.org/pub/release-104/fasta/canis_lupus_familiaris/pep/ |
| Protein language model | **ESM-2 15B** (`esm2_t48_15B_UR50D`) | https://huggingface.co/facebook/esm2_t48_15B_UR50D |
| UCE model & code | `snap-stanford/UCE` (+ figshare model files 24320806) | https://github.com/snap-stanford/UCE |
| Authors' labelled UCE map | Tabula Sapiens UCE embeddings (Zenodo) | https://doi.org/10.5281/zenodo.19462110 |
| Human reference atlas | CZ CELLxGENE Census | https://chanzuckerberg.github.io/cellxgene-census/ |
| Baseline tool | CellTypist (pretrained immune models) | https://www.celltypist.org/ |
| Compute | RunPod cloud GPU (A40 / RTX 4090) | https://www.runpod.io/ |
| **Our dog ESM-2 embeddings** | 20,257 genes × 5120-d, ~201 MB | see [`data/DATA.md`](data/DATA.md) |

## Notes / gotchas learned along the way

- **figshare from a datacenter**: use `https://ndownloader.figshare.com/files/{id}` (the `figshare.com/ndownloader/...` host blocks datacenter IPs).
- **ESM-2 15B**: pin `transformers==4.44.2`; load with `device_map={"":0}`; ~60.5 GB fp32.
- **Cross-species embeddings**: subtract the per-species mean (offset ≈ 0.6–0.7) before transfer; L2-normalise for cosine kNN.
- **Zenodo big files** throttle from datacenters — resume with `curl -C -` + `--speed-limit`, or download only the needed tar member via HTTP range.
- **Census v2023-12-15** has Tabula Sapiens v1 (483k cells), not TS 2.0.
