"""Resumably download only the FIRST member of the Zenodo other_files.tar.gz
(the authors' Tabula Sapiens UCE-embedded atlas) via HTTP Range, extract it,
and stop. Writes progress to PART.progress."""
import urllib.request, os, tarfile, sys, time

URL = "https://zenodo.org/api/records/19462110/files/other_files.tar.gz/content"
PART = sys.argv[1] if len(sys.argv) > 1 else '/tmp/part.gz'
OUTH5 = sys.argv[2] if len(sys.argv) > 2 else '/tmp/member1.h5ad'
MEMBER = 'tabula_sapiens_subset_shuffled_uce_adata.h5ad'
PROG = PART + '.progress'

def dsize():
    return os.path.getsize(PART) if os.path.exists(PART) else 0

def log(m):
    open(PROG, 'w').write(f"{time.strftime('%H:%M:%S')} {m}\n")
    print(m, flush=True)

def download_more(nbytes):
    start = dsize()
    req = urllib.request.Request(URL, headers={'Range': f'bytes={start}-{start+nbytes-1}'})
    with urllib.request.urlopen(req, timeout=120) as r:
        code = r.status
        if code == 200 and start > 0:
            # server ignored Range -> restart clean
            open(PART, 'wb').close(); start = 0
        with open(PART, 'ab') as w:
            got = 0
            while True:
                c = r.read(1 << 20)
                if not c:
                    break
                w.write(c); got += len(c)
                if got % (64 << 20) < (1 << 20):
                    log(f"downloaded {dsize()//1024//1024}MB (+{got//1024//1024}MB this pull)")
        return got

def try_extract():
    try:
        with open(PART, 'rb') as f, tarfile.open(fileobj=f, mode='r|gz') as tar:
            for m in tar:
                if os.path.basename(m.name) == MEMBER:
                    with open(OUTH5, 'wb') as w:
                        src = tar.extractfile(m)
                        while True:
                            c = src.read(1 << 24)
                            if not c:
                                break
                            w.write(c)
                    return True
                break
        return False
    except (EOFError, tarfile.ReadError, OSError) as ex:
        return False

log("start")
CHUNK = 512 << 20  # 512MB per pull, then test-extract
attempt = 0
while True:
    if try_extract():
        log(f"EXTRACTED {OUTH5} {os.path.getsize(OUTH5)//1024//1024}MB after {dsize()//1024//1024}MB gz")
        break
    attempt += 1
    try:
        got = download_more(CHUNK)
        log(f"pull#{attempt} got {got//1024//1024}MB total={dsize()//1024//1024}MB")
        if got == 0:
            log("server returned 0 bytes; sleeping"); time.sleep(10)
    except Exception as ex:
        log(f"pull#{attempt} error {ex!r}; retrying"); time.sleep(8)
