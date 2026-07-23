import json, base64, urllib.request, urllib.error
KEY = open('/root/.rpkey').read().strip()
VOL = open('/root/.rp_volid').read().strip()

def b64(path):
    return base64.b64encode(open(path,'rb').read()).decode()

sc_parse   = b64('repro/01_parse_proteome.py')
sc_dogh5ad = b64('build_dog_h5ad_enscafg.py')
sc_ref     = b64('build_ref_pod.py')
sc_newspec = b64('build_newspecies.py')
sc_extract = b64('extract_uce.py')

ENS  = "https://ftp.ensembl.org/pub/release-104/fasta/canis_lupus_familiaris/pep/Canis_lupus_familiaris.CanFam3.1.pep.all.fa.gz"
GEO  = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE225nnn/GSE225599/suppl/GSE225599_RAW.tar"
META = "https://cells.ucsc.edu/canine-leukocyte-atlas/healthy-os-combined/meta.tsv"

startup = f"""#!/bin/bash
mkdir -p /workspace/out /workspace/work
cd /workspace/work
printf '%s' '{sc_parse}'   | base64 -d > 01_parse_proteome.py
printf '%s' '{sc_dogh5ad}' | base64 -d > build_dog_h5ad_enscafg.py
printf '%s' '{sc_ref}'     | base64 -d > build_ref_pod.py
printf '%s' '{sc_newspec}' | base64 -d > build_newspecies.py
printf '%s' '{sc_extract}' | base64 -d > extract_uce.py
cd /workspace/out && nohup python -m http.server 8000 >/dev/null 2>&1 &
pip install -q jupyterlab >/dev/null 2>&1
nohup jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser --ServerApp.token=doguce --ServerApp.root_dir=/workspace --ServerApp.disable_check_xsrf=True >/dev/null 2>&1 &
cd /workspace/work
exec >> /workspace/out/run.log 2>&1
S=/workspace/out/status.txt
st(){{ echo "STATUS=$1 $(date -u)" > $S; echo "=== $1 $(date -u) ==="; }}

st SETUP
python -c "import torch;print('torch',torch.__version__,'GPU',torch.cuda.get_device_name(0))"
pip install -q "transformers==4.44.2" "huggingface_hub<0.26" "accelerate>=0.33" scanpy anndata cellxgene-census scikit-misc numpy pandas scipy || {{ st ERROR_DEPS; tail -f /dev/null; }}

[ -d /workspace/UCE ] || git clone --depth 1 https://github.com/snap-stanford/UCE.git /workspace/UCE
cd /workspace/UCE && mkdir -p model_files out

st MODEL_FILES
dl(){{ [ -s "$2" ] && {{ echo "cached $2"; return; }}; for k in 1 2 3 4 5 6; do curl -sS -L -A Mozilla/5.0 --max-time 3600 -o "$2" "$1" && [ -s "$2" ] && break; echo "retry $2 ($k)"; sleep 15; done; echo "dl $2 -> $(wc -c < $2 2>/dev/null) bytes"; }}
dl https://ndownloader.figshare.com/files/42706558 model_files/species_chrom.csv
dl https://ndownloader.figshare.com/files/42706555 model_files/species_offsets.pkl
dl https://ndownloader.figshare.com/files/42706585 model_files/all_tokens.torch
dl https://ndownloader.figshare.com/files/43423236 model_files/33l_8ep_1024t_1280.torch
if [ ! -d model_files/protein_embeddings ]; then dl https://ndownloader.figshare.com/files/42715213 model_files/protein_embeddings.tar.gz; tar -xzf model_files/protein_embeddings.tar.gz -C model_files/; fi
# sanity: model files must be non-trivial
for f in model_files/all_tokens.torch model_files/33l_8ep_1024t_1280.torch; do [ "$(wc -c < $f)" -lt 1000000 ] && {{ st ERROR_MODELFILES; tail -f /dev/null; }}; done

st DOG_PROTEOME
if [ ! -f /workspace/dog_longest.pep.fa ]; then
  cd /workspace/work; curl -sS -L -o cf31.fa.gz {ENS!r}; python 01_parse_proteome.py --fasta cf31.fa.gz --outdir /workspace; cd /workspace/UCE
fi

st DOG_H5AD
if [ ! -f /workspace/dog_eval_enscafg.h5ad ]; then
  cd /workspace/work; rm -rf /workspace/work/dogmtx
  [ -s /workspace/GSE225599_RAW.tar ] || curl -sS -L --retry 6 --retry-delay 10 --max-time 1800 -o /workspace/GSE225599_RAW.tar {GEO!r}
  python build_dog_h5ad_enscafg.py --raw-tar /workspace/GSE225599_RAW.tar --out /workspace/dog_eval_enscafg.h5ad --workdir /workspace/work/dogmtx || {{ st ERROR_DOGH5AD; tail -f /dev/null; }}
  cd /workspace/UCE
fi

st HUMAN_REF
[ -f /workspace/human_blood_immune_ref.h5ad ] || REF_OUT=/workspace/human_blood_immune_ref.h5ad python /workspace/work/build_ref_pod.py || {{ st ERROR_HUMANREF; tail -f /dev/null; }}

st WAITING_FOR_PT
while :; do
  f=$(find /workspace -maxdepth 3 -name 'dog_esm2_embeddings.pt' 2>/dev/null | head -1)
  if [ -n "$f" ] && [ "$(wc -c < "$f" 2>/dev/null)" = "209911041" ]; then cp "$f" /workspace/dog_esm2_embeddings.pt; break; fi
  sleep 15
done
st GOT_PT

st NEWSPECIES
python /workspace/work/build_newspecies.py --all-tokens model_files/all_tokens.torch --pe /workspace/dog_esm2_embeddings.pt --gene-table /workspace/dog_gene_table.csv --outdir model_files || {{ st ERROR_NEWSPECIES; tail -f /dev/null; }}
grep -q '^species,path' model_files/new_species_protein_embeddings.csv || echo 'species,path' > model_files/new_species_protein_embeddings.csv
grep -q '^dog,' model_files/new_species_protein_embeddings.csv || echo 'dog,/workspace/dog_esm2_embeddings.pt' >> model_files/new_species_protein_embeddings.csv

st UCE_DOG
python eval_single_anndata.py --adata_path /workspace/dog_eval_enscafg.h5ad --species dog --nlayers 33 --model_loc model_files/33l_8ep_1024t_1280.torch --CHROM_TOKEN_OFFSET 20261 --spec_chrom_csv_path model_files/dog_to_chrom_pos.csv --token_file model_files/dog_pe_tokens.torch --offset_pkl_path model_files/dog_offsets.pkl --dir /workspace/out/ --batch_size 25 || {{ st ERROR_UCE_DOG; tail -f /dev/null; }}

st UCE_HUMAN
python eval_single_anndata.py --adata_path /workspace/human_blood_immune_ref.h5ad --species human --nlayers 33 --model_loc model_files/33l_8ep_1024t_1280.torch --dir /workspace/out/ --batch_size 25 || {{ st ERROR_UCE_HUMAN; tail -f /dev/null; }}

st EXTRACT
python /workspace/work/extract_uce.py /workspace/out/dog_eval_enscafg_uce_adata.h5ad /workspace/out/dog || {{ st ERROR_EXTRACT; tail -f /dev/null; }}
python /workspace/work/extract_uce.py /workspace/out/human_blood_immune_ref_uce_adata.h5ad /workspace/out/human || {{ st ERROR_EXTRACT2; tail -f /dev/null; }}
st DONE
tail -f /dev/null
"""

payload = {
    "name": "dog-uce-inference",
    "cloudType": "SECURE",
    "computeType": "GPU",
    "gpuTypeIds": ["NVIDIA A40","NVIDIA RTX A6000","NVIDIA L40","NVIDIA L40S","NVIDIA GeForce RTX 4090","NVIDIA RTX A5000","NVIDIA A100 80GB PCIe"],
    "gpuTypePriority": "availability",
    "gpuCount": 1,
    "minRAMPerGPU": 24,
    "containerDiskInGb": 40,
    "volumeMountPath": "/workspace",
    "networkVolumeId": VOL,
    "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
    "ports": ["8000/http","8888/http"],
    "dockerStartCmd": ["bash","-c",startup],
    "env": {"HF_HUB_DISABLE_TELEMETRY":"1"},
}
req = urllib.request.Request("https://rest.runpod.io/v1/pods",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type":"application/json"}, method="POST")
try:
    with urllib.request.urlopen(req, timeout=90) as r:
        resp=json.loads(r.read()); pid=resp.get("id")
        print("CREATED pod", pid, "status", resp.get("desiredStatus"))
        open('/root/.rp_podid','w').write(pid or "")
        print("logs:   https://%s-8000.proxy.runpod.net/status.txt" % pid)
        print("upload: https://%s-8888.proxy.runpod.net/?token=doguce" % pid)
except urllib.error.HTTPError as e:
    print("HTTP", e.code, e.read().decode()[:600])
