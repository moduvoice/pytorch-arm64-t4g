#!/usr/bin/env python3
"""Build the PEP 503 simple index (docs/simple/) from wheels.json.

The index only links to the wheels; it does not host them. Every link points at
the public AWS `framework-binaries` bucket where AWS published these builds.

    python3 tools/build_index.py
"""
import html
import json
import pathlib
from urllib.parse import quote

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASE = "https://framework-binaries.s3.us-west-2.amazonaws.com/"


def page(title, links):
    body = "\n".join(f'    <a href="{html.escape(h)}">{html.escape(t)}</a><br/>' for t, h in links)
    return (f"<!DOCTYPE html>\n<html>\n  <head><meta name=\"pypi:repository-version\" content=\"1.0\">"
            f"<title>{html.escape(title)}</title></head>\n  <body>\n{body}\n  </body>\n</html>\n")


def main():
    wheels = json.loads((ROOT / "wheels.json").read_text(encoding="utf-8"))
    out = ROOT / "docs" / "simple"
    projects = {}
    for w in wheels:
        if w["sm_75"]:
            name = w["file"].split("-")[0]
            projects.setdefault(name, []).append(w)
    for name, ws in projects.items():
        links = [(w["file"], BASE + quote(w["key"]) + "#sha256=" + w["sha256"]) for w in ws]
        (out / name).mkdir(parents=True, exist_ok=True)
        (out / name / "index.html").write_text(page(f"Links for {name}", links), encoding="utf-8")
    root_links = [(n, f"{n}/") for n in sorted(projects)]
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(page("Simple index", root_links), encoding="utf-8")
    print({n: len(ws) for n, ws in projects.items()})


if __name__ == "__main__":
    main()
