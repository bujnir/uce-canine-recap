# Data artifacts

## Dog ESM-2 protein embeddings (the artifact we generated)

- **File:** `dog_esm2_embeddings.pt`
- **Contents:** a dict `{ ENSCAFG_gene_id : float16[5120] }` — one ESM-2 (15B) protein embedding per dog gene
- **Genes:** 20,257 (longest protein per gene, Ensembl CanFam3.1 / release-104)
- **Size:** ~201 MB · md5 `7c35a83b285dd4163c445fb6de2ab364`

GitHub rejects files > 100 MB in normal storage, so this artifact is **hosted outside the git tree**.

### Recommended: Hugging Face Hub (best home for a reusable ML artifact)

```python
from huggingface_hub import hf_hub_download
import torch

path = hf_hub_download(repo_id="<your-username>/dog-esm2-15b-embeddings",
                       filename="dog_esm2_embeddings.pt",
                       repo_type="dataset")
emb = torch.load(path)          # dict: ENSCAFG -> float16[5120]
print(len(emb), next(iter(emb.values())).shape)   # 20257, torch.Size([5120])
```

To publish (once):

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli repo create dog-esm2-15b-embeddings --type dataset
huggingface-cli upload <your-username>/dog-esm2-15b-embeddings dog_esm2_embeddings.pt --repo-type dataset
```

Add a short dataset card (`README.md`) noting: model `esm2_t48_15B_UR50D`, mean-pooled, fp16, 20,257 dog
genes keyed by Ensembl ID, produced for the UCE canine recap.

### Alternative: GitHub Release asset

If you prefer to keep everything on GitHub, attach the `.pt` to a Release (assets can be up to 2 GB):

```bash
gh release create v1.0 dog_esm2_embeddings.pt \
  --title "Dog ESM-2 15B protein embeddings" \
  --notes "20,257 dog genes x 5120-d, fp16. md5 7c35a83b285dd4163c445fb6de2ab364"
```

Then link it from the README. (Do **not** commit the raw file into the repo tree.)

## Regenerating it from scratch

```bash
python ../scripts/01_parse_proteome.py --fasta Canis_lupus_familiaris.CanFam3.1.pep.all.fa.gz --outdir .
python ../scripts/embed_esm2.py --fasta dog_longest.pep.fa --out dog_esm2_embeddings.pt --token-budget 4096 --resume
```

Needs a GPU with ~40 GB VRAM for ESM-2 15B (we used a RunPod A40, ~75 min).
