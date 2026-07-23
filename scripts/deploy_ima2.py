import json, base64, urllib.request, urllib.error
KEY=open('/root/.rpkey').read().strip()
def b64(p): return base64.b64encode(open(p,'rb').read()).decode()
sc=b64('centroid_from_tar.py')
TAR="https://zenodo.org/api/records/19462110/files/other_files.tar.gz/content"
EXP=66031481459

startup=f"""#!/bin/bash
mkdir -p /workspace/out /workspace/work; cd /workspace/work
printf '%s' '{sc}' | base64 -d > centroid_from_tar.py
cd /workspace/out && nohup python -m http.server 8000 >/dev/null 2>&1 &
cd /workspace/work; exec >> /workspace/out/run.log 2>&1
S=/workspace/out/status.txt; st(){{ echo "STATUS=$1 $(date -u)" > $S; echo "=== $1 $(date -u) ==="; }}
TARF=/workspace/other.tar.gz; EXP={EXP}
# background progress logger
( while true; do sz=$(stat -c%s $TARF 2>/dev/null || echo 0); echo "$(date -u +%H:%M:%S) dl=$sz/$EXP ($((sz*100/EXP))%)" > /workspace/out/progress.txt; sleep 20; done ) &
st SETUP
pip install -q h5py numpy || {{ st ERROR_DEPS; tail -f /dev/null; }}
st DOWNLOAD
for k in $(seq 1 60); do
  sz=$(stat -c%s $TARF 2>/dev/null || echo 0)
  [ "$sz" -ge "$EXP" ] && break
  echo "dl attempt $k from $sz"
  curl -sS -L -C - --retry 5 --retry-delay 10 --retry-all-errors --speed-time 60 --speed-limit 300000 --max-time 3600 -o $TARF "{TAR}"
  sleep 5
done
sz=$(stat -c%s $TARF 2>/dev/null || echo 0)
echo "final size $sz / $EXP"
[ "$sz" -ge "$EXP" ] || {{ st ERROR_DOWNLOAD; tail -f /dev/null; }}
st CENTROIDS
python /workspace/work/centroid_from_tar.py $TARF /workspace/out /workspace/tmp_member.h5ad || {{ st ERROR_CENTROIDS; tail -f /dev/null; }}
# free the big tarball
rm -f $TARF
st DONE
tail -f /dev/null
"""
payload={"name":"ima-centroids2","cloudType":"SECURE","computeType":"GPU",
 "gpuTypeIds":["NVIDIA RTX A4000","NVIDIA RTX A4500","NVIDIA RTX A5000","NVIDIA A40","NVIDIA RTX A6000","NVIDIA GeForce RTX 4090","NVIDIA L4","NVIDIA L40","NVIDIA L40S","NVIDIA RTX 4000 Ada Generation","NVIDIA RTX 2000 Ada Generation","NVIDIA GeForce RTX 3090"],
 "gpuTypePriority":"availability","gpuCount":1,"minRAMPerGPU":16,
 "containerDiskInGb":25,"volumeInGb":100,"volumeMountPath":"/workspace",
 "imageName":"runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
 "ports":["8000/http"],"dockerStartCmd":["bash","-c",startup],"env":{"HF_HUB_DISABLE_TELEMETRY":"1"}}
req=urllib.request.Request("https://rest.runpod.io/v1/pods",data=json.dumps(payload).encode(),
    headers={"Authorization":f"Bearer {KEY}","Content-Type":"application/json"},method="POST")
try:
    r=urllib.request.urlopen(req,timeout=90); d=json.loads(r.read()); pid=d.get("id")
    print("CREATED",pid); open('/root/.rp_podid_ima','w').write(pid or "")
    print("status:  https://%s-8000.proxy.runpod.net/status.txt"%pid)
    print("progress:https://%s-8000.proxy.runpod.net/progress.txt"%pid)
    print("summary: https://%s-8000.proxy.runpod.net/ima_summary.json"%pid)
except urllib.error.HTTPError as e: print("HTTP",e.code,e.read().decode()[:400])
