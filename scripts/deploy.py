import json, base64, urllib.request, urllib.error
KEY = open('/root/.rpkey').read().strip()

emb = open('runpod/embed_esm2.py','rb').read()
par = open('repro/01_parse_proteome.py','rb').read()
b_emb = base64.b64encode(emb).decode()
b_par = base64.b64encode(par).decode()

ENS = ("https://ftp.ensembl.org/pub/release-104/fasta/canis_lupus_familiaris/"
       "pep/Canis_lupus_familiaris.CanFam3.1.pep.all.fa.gz")

# NOTE: no `set -e` inside a || group (bash neuters it). Explicit || per step.
startup = f"""#!/bin/bash
mkdir -p /workspace/out /workspace/hf
cd /workspace
printf '%s' '{b_par}' | base64 -d > 01_parse_proteome.py
printf '%s' '{b_emb}' | base64 -d > embed_esm2.py
cd /workspace/out && nohup python -m http.server 8000 >/dev/null 2>&1 &
cd /workspace
exec >> /workspace/out/run.log 2>&1
S=/workspace/out/status.txt
echo "STATUS=STARTED $(date -u)" > $S
python -c "import torch;print('torch',torch.__version__,'| GPU',torch.cuda.get_device_name(0))"
echo "== deps =="
pip install -q "transformers==4.44.2" "accelerate>=0.33" "huggingface_hub<0.26" numpy || {{ echo "STATUS=ERROR_DEPS $(date -u)" > $S; tail -f /dev/null; }}
python -c "import transformers,torch;print('transformers',transformers.__version__,'torch',torch.__version__)"
echo "STATUS=DEPS_OK $(date -u)" > $S
echo "== fetch Ensembl proteome =="
python -c "import urllib.request; urllib.request.urlretrieve('{ENS}','cf31.pep.all.fa.gz'); print('downloaded')" || {{ echo "STATUS=ERROR_FETCH $(date -u)" > $S; tail -f /dev/null; }}
python 01_parse_proteome.py --fasta cf31.pep.all.fa.gz --outdir . || {{ echo "STATUS=ERROR_PARSE $(date -u)" > $S; tail -f /dev/null; }}
echo "FASTA md5:"; md5sum dog_longest.pep.fa
echo "STATUS=FASTA_OK $(date -u)" > $S
echo "== smoke test (20 seqs) =="
python embed_esm2.py --fasta dog_longest.pep.fa --out /workspace/out/smoke.pt --limit 20 || {{ echo "STATUS=ERROR_SMOKE $(date -u)" > $S; tail -f /dev/null; }}
echo "STATUS=SMOKE_OK $(date -u)" > $S
echo "== full embedding run =="
echo "STATUS=EMBEDDING $(date -u)" > $S
python embed_esm2.py --fasta dog_longest.pep.fa --out /workspace/out/dog_esm2_embeddings.pt --token-budget 4096 --resume || {{ echo "STATUS=ERROR_FULL $(date -u)" > $S; tail -f /dev/null; }}
echo "STATUS=DONE $(date -u)" > $S
tail -f /dev/null
"""

payload = {
    "name": "dog-esm2-embed",
    "cloudType": "SECURE",
    "computeType": "GPU",
    "gpuTypeIds": ["NVIDIA A40", "NVIDIA RTX A6000", "NVIDIA L40", "NVIDIA L40S",
                   "NVIDIA RTX 6000 Ada Generation", "NVIDIA A100 80GB PCIe"],
    "gpuTypePriority": "availability",
    "gpuCount": 1,
    "minRAMPerGPU": 24,
    "containerDiskInGb": 100,
    "volumeInGb": 20,
    "volumeMountPath": "/workspace",
    "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
    "ports": ["8000/http"],
    "dockerStartCmd": ["bash", "-c", startup],
    "env": {"HF_HUB_DISABLE_TELEMETRY": "1"},
}

req = urllib.request.Request("https://rest.runpod.io/v1/pods",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type":"application/json"},
        method="POST")
try:
    with urllib.request.urlopen(req, timeout=90) as r:
        resp = json.loads(r.read())
        pid = resp.get("id")
        print("CREATED  pod id:", pid, "| status:", resp.get("desiredStatus"))
        open('/root/.rp_podid','w').write(pid or "")
        print("proxy base: https://%s-8000.proxy.runpod.net" % pid)
except urllib.error.HTTPError as e:
    print("HTTP", e.code, e.read().decode()[:800])
