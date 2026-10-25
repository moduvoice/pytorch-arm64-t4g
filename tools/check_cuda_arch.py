#!/usr/bin/env python3
"""List the CUDA architectures compiled into a PyTorch wheel - no GPU needed.

Reads the fatbinary headers embedded in the shared libraries inside the wheel and
prints which SASS (sm_XX) and PTX (compute_XX) targets are present.
Works on any host (x86_64 or aarch64) for any wheel.

    python3 check_cuda_arch.py torch-2.11.0+cu130-cp313-cp313-manylinux_2_28_aarch64.whl
"""
import mmap
import os
import struct
import sys
import tempfile
import zipfile

FATBIN_MAGIC = 0xBA55ED50
KIND = {1: "ptx", 2: "sass"}


def scan(path):
    """Return {"sass": set(arch), "ptx": set(arch)} for one ELF file."""
    found = {"sass": set(), "ptx": set()}
    with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
        pos = m.find(struct.pack("<I", FATBIN_MAGIC))
        while pos != -1:
            version, hsize = struct.unpack_from("<HH", m, pos + 4)
            if version == 1 and hsize == 16:
                (fsize,) = struct.unpack_from("<Q", m, pos + 8)
                e, end = pos + hsize, min(pos + hsize + fsize, len(m))
                while e + 32 <= end:
                    kind, _, ehsize, esize = struct.unpack_from("<HHIQ", m, e)
                    (arch,) = struct.unpack_from("<I", m, e + 28)
                    if kind not in KIND or ehsize < 32:
                        break
                    found[KIND[kind]].add(arch)
                    e += ehsize + esize
                nxt = max(end, pos + 4)
            else:
                nxt = pos + 4
            pos = m.find(struct.pack("<I", FATBIN_MAGIC), nxt)
    return found


def fmt(r):
    sass = " ".join(f"sm_{a}" for a in sorted(r["sass"]))
    ptx = " ".join(f"compute_{a}" for a in sorted(r["ptx"]))
    return f"SASS: {sass or '-'} | PTX: {ptx or '-'}"


def main(wheel):
    """Per-library listing. What matters is torch's own kernels in libtorch_cuda.so;
    bundled NVIDIA libraries (cuBLAS, cuDNN, ...) carry their own wide arch sets."""
    print(os.path.basename(wheel))
    core = None
    with zipfile.ZipFile(wheel) as z, tempfile.TemporaryDirectory() as tmp:
        libs = [n for n in z.namelist() if n.endswith(".so") or ".so." in n]
        for name in sorted(libs):
            p = z.extract(name, tmp)
            r = scan(p)
            os.remove(p)
            if r["sass"] or r["ptx"]:
                print(f"  {name}: {fmt(r)}")
            if name.endswith("/libtorch_cuda.so"):
                core = r
    if core is not None:
        print(f"  sm_75 in libtorch_cuda.so (T4/T4G): {'YES' if 75 in core['sass'] else 'NO'}")


def summary(wheel):
    """{"libs": {name: {"sass": [...], "ptx": [...]}}} for libraries that carry CUDA code."""
    libs = {}
    with zipfile.ZipFile(wheel) as z, tempfile.TemporaryDirectory() as tmp:
        for name in sorted(n for n in z.namelist() if n.endswith(".so") or ".so." in n):
            p = z.extract(name, tmp)
            r = scan(p)
            os.remove(p)
            if r["sass"] or r["ptx"]:
                libs[name] = {k: sorted(v) for k, v in r.items()}
    return {"libs": libs}


if __name__ == "__main__":
    if sys.argv[1:2] == ["--json"]:
        import json
        for w in sys.argv[2:]:
            print(json.dumps({"file": os.path.basename(w), **summary(w)}))
    else:
        for w in sys.argv[1:]:
            main(w)
