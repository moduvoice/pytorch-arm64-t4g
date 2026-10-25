#!/usr/bin/env python3
"""Turn check results (JSON lines from `check_cuda_arch.py --json` + key/sha256) into wheels.json.

A torch wheel is marked sm_75 only if torch's own kernels (torch/lib/libtorch_cuda.so)
contain sm_75 SASS - bundled NVIDIA libraries are ignored. torchvision/torchaudio are
marked sm_75 if their torch partner passes and every CUDA library they ship has sm_75
(or they ship no CUDA code at all).

    python3 tools/make_wheels_json.py results.jsonl > wheels.json
"""
import json
import re
import sys

rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.startswith("{")]
bad = [r for r in rows if "error" in r]
if bad:
    sys.exit(f"download errors: {bad}")

out = []
for r in rows:
    name, ver, py = re.match(r"([a-z]+)-([^-]+)-(cp\d+)-", r["file"]).groups()
    libs = r["libs"]
    if name == "torch":
        core = [v for k, v in libs.items() if k.endswith("/libtorch_cuda.so")]
        ok = bool(core) and 75 in core[0]["sass"]
    else:
        ok = all(75 in v["sass"] for v in libs.values())
    out.append({"file": r["file"], "key": r["key"], "sha256": r["sha256"], "project": name,
                "version": ver, "python": py, "torch_release": r["key"].split("/")[1], "sm_75": ok})

# torchvision/torchaudio follow the torch wheel of the same release folder
torch_ok = {(w["torch_release"], w["python"]) for w in out if w["project"] == "torch" and w["sm_75"]}
for w in out:
    if w["project"] != "torch":
        w["sm_75"] = w["sm_75"] and (w["torch_release"], w["python"]) in torch_ok
print(json.dumps(out, indent=1))
