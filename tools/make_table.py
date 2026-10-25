#!/usr/bin/env python3
"""Write the wheel table from wheels.json into README.md / README.ko.md (between TABLE markers)."""
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
wheels = [w for w in json.loads((ROOT / "wheels.json").read_text(encoding="utf-8")) if w["sm_75"]]


def vkey(v):
    return tuple(int(x) for x in re.match(r"[\d.]+", v).group().rstrip(".").split("."))


rows = []
for t in sorted((w for w in wheels if w["project"] == "torch"), key=lambda w: vkey(w["version"]), reverse=True):
    cu = t["version"].split("+")[1]
    same = [w for w in wheels if w["torch_release"] == t["torch_release"] and w["python"] == t["python"]
            and w["version"].endswith("+" + cu)]
    tv = next((w["version"] for w in same if w["project"] == "torchvision"), "—")
    ta = next((w["version"] for w in same if w["project"] == "torchaudio"), "—")
    py = f"3.{t['python'][3:]}"
    rows.append(f"| `{t['version']}` | {cu[2:-1]}.{cu[-1]} | {py} | `{tv}` | `{ta}` |")

HEAD = {
    "README.md": "| torch | CUDA | Python | torchvision | torchaudio |\n|---|---|---|---|---|",
    "README.ko.md": "| torch | CUDA | 파이썬 | torchvision | torchaudio |\n|---|---|---|---|---|",
}
for name, head in HEAD.items():
    p = ROOT / name
    text = p.read_text(encoding="utf-8")
    block = "<!-- TABLE -->\n" + head + "\n" + "\n".join(rows) + "\n<!-- /TABLE -->"
    text = re.sub(r"<!-- TABLE -->(.*?<!-- /TABLE -->)?", block, text, count=1, flags=re.S)
    p.write_text(text, encoding="utf-8", newline="\n")
print(len(rows), "rows")
