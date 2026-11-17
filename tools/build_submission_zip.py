"""Rebuild the uploaded test-phase archive byte for byte from submission/.

usage: python tools/build_submission_zip.py [out.zip]
Fixed timestamps (2026-11-13 00:00), sorted names, LF line endings, mode 644.
Prints the SHA-256 and checks it against the uploaded archive's hash.
"""
import os, sys, zipfile, hashlib

EXPECTED = '07e4ba37ac77bc2d78b8695ba0a984cc21df74ba61a7de02395cf29b59c76f75'
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'submission')
out = sys.argv[1] if len(sys.argv) > 1 else 'SimulacraBench_F5_KumoStackRobust.zip'
names = sorted(os.listdir(root))
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in names:
        data = open(os.path.join(root, f), 'rb').read().replace(b'\r\n', b'\n')
        zi = zipfile.ZipInfo(f, (2026, 11, 13, 0, 0, 0))
        zi.compress_type = zipfile.ZIP_DEFLATED
        zi.external_attr = 0o644 << 16
        z.writestr(zi, data)
h = hashlib.sha256(open(out, 'rb').read()).hexdigest()
print(out, os.path.getsize(out), h)
print('MATCHES uploaded archive' if h == EXPECTED else 'DIFFERS from uploaded archive (zlib version can change bytes; compare file contents instead)')
sys.exit(0 if h == EXPECTED else 1)
