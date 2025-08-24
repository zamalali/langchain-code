from __future__ import annotations
from pathlib import Path
from typing import Any, Optional, Iterable, Tuple
import json
import time
import re
import os
import base64
import zlib
import shutil
import tempfile
import subprocess

from langchain_core.tools import tool

_MERMAID_OK = True
try:
    from langchain_core.runnables.graph_mermaid import (
        draw_mermaid,
        draw_mermaid_png,
        MermaidDrawMethod,
        CurveStyle,
    )
    try:
        from langchain_core.runnables.graph_mermaid import NodeStyles  # type: ignore
        _HAS_NODESTYLES = True
    except Exception:
        _HAS_NODESTYLES = False
except Exception:
    _MERMAID_OK = False
    _HAS_NODESTYLES = False

# Optional YAML support if present
try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False


def _rooted(project_dir: str, path: str) -> Path:
    p = Path(project_dir).joinpath(path or "").resolve()
    if not str(p).startswith(str(Path(project_dir).resolve())):
        raise ValueError("Path escapes project_dir")
    return p


def _loads_json_or_yaml(val: Any, default: Any):
    if val is None:
        return default
    if isinstance(val, (dict, list)):
        return val
    if isinstance(val, str) and val.strip():
        s = val.strip()
        try:
            return json.loads(s)
        except Exception:
            pass
        if _HAS_YAML:
            try:
                yv = yaml.safe_load(s)
                if isinstance(yv, (dict, list)):
                    return yv
            except Exception:
                pass
    return default


def _looks_like_mermaid(src: Any) -> Optional[str]:
    if not isinstance(src, str):
        return None
    s = src.strip()
    if re.search(r"\b(graph|flowchart|sequenceDiagram|classDiagram|stateDiagram|erDiagram|gantt)\b", s):
        m = re.search(r"```(?:mermaid)?\s*(.*?)```", s, flags=re.S | re.I)
        return m.group(1).strip() if m else s
    return None


def _norm_nodes(raw: Any) -> dict[str, str] | str:
    m = _looks_like_mermaid(raw)
    if m is not None:
        return m

    raw = _loads_json_or_yaml(raw, {})
    nodes: dict[str, str] = {}

    if isinstance(raw, dict):
        if all(isinstance(v, (str, int, float)) for v in raw.values()):
            return {str(k): str(v) for k, v in raw.items()}
        return {}

    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                nid = str(item.get("id") or item.get("name") or item.get("key") or item.get("node") or "").strip()
                if not nid:
                    if len(item) == 1:
                        k, v = next(iter(item.items()))
                        nid = str(k)
                        nodes[nid] = str(v)
                        continue
                    continue
                label = str(item.get("label") or item.get("title") or item.get("name") or nid)
                nodes[nid] = label
            else:
                nid = str(item)
                nodes[nid] = nid
        return nodes

    return {}


def _parse_edge_string(s: str) -> Optional[dict]:
    s = s.strip()
    m = re.match(r"^\s*([\w.\-:/]+)\s*[-=]{1,3}>\s*([\w.\-:/]+)\s*[:|]?\s*(.*)$", s)
    if not m:
        return None
    src, tgt, lbl = m.group(1), m.group(2), m.group(3).strip()
    data = {"label": lbl} if lbl else {}
    return {"source": str(src), "target": str(tgt), "data": data}


def _flatten(iterable: Iterable[Any]) -> list[Any]:
    return [it for it in iterable]


def _norm_edges(raw: Any) -> tuple[list[dict], dict[str, str], Optional[str]]:
    m = _looks_like_mermaid(raw)
    if m is not None:
        return ([], {}, m)

    raw = _loads_json_or_yaml(raw, [])
    edges: list[dict] = []
    inferred_nodes: dict[str, str] = {}

    if isinstance(raw, dict):
        for k, v in raw.items():
            src = str(k)
            if isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, dict):
                        tgt = item.get("target") or item.get("to") or item.get("dst") or item.get("node")
                        if not tgt:
                            continue
                        lbl = item.get("label")
                        data = {"label": lbl} if lbl is not None else {}
                        edges.append({"source": src, "target": str(tgt), "data": data})
                        inferred_nodes[src] = src
                        inferred_nodes[str(tgt)] = str(tgt)
                    else:
                        tgt = str(item)
                        edges.append({"source": src, "target": tgt, "data": {}})
                        inferred_nodes[src] = src
                        inferred_nodes[tgt] = tgt
            elif isinstance(v, str):
                e = _parse_edge_string(f"{k}->{v}")
                if e:
                    edges.append(e)
                    inferred_nodes[str(k)] = str(k)
                    inferred_nodes[str(v)] = str(v)
        return (edges, inferred_nodes, None)

    if isinstance(raw, list):
        for e in _flatten(raw):
            if isinstance(e, dict):
                src = e.get("source") or e.get("from") or e.get("src")
                tgt = e.get("target") or e.get("to") or e.get("dst")
                if not (src and tgt):
                    if len(e) == 1:
                        k, v = next(iter(e.items()))
                        src, tgt = k, v
                    else:
                        continue
                lbl = e.get("label")
                data = e.get("data")
                if not isinstance(data, dict):
                    data = {"label": lbl} if lbl is not None else {}
                src_s, tgt_s = str(src), str(tgt)
                edges.append({"source": src_s, "target": tgt_s, "data": data})
                inferred_nodes[src_s] = src_s
                inferred_nodes[tgt_s] = tgt_s
            elif isinstance(e, (list, tuple)) and len(e) >= 2:
                src, tgt = e[0], e[1]
                lbl = e[2] if len(e) >= 3 else None
                src_s, tgt_s = str(src), str(tgt)
                data = {"label": lbl} if lbl is not None else {}
                edges.append({"source": src_s, "target": tgt_s, "data": data})
                inferred_nodes[src_s] = src_s
                inferred_nodes[tgt_s] = tgt_s
            elif isinstance(e, str):
                parsed = _parse_edge_string(e)
                if parsed:
                    edges.append(parsed)
                    inferred_nodes[parsed["source"]] = parsed["source"]
                    inferred_nodes[parsed["target"]] = parsed["target"]
        return (edges, inferred_nodes, None)

    return ([], {}, None)


# ---------------- HTTP helpers & API renderers ----------------

def _http_get(url: str, timeout: int = 30) -> Tuple[bytes, str]:
    import urllib.request
    req = urllib.request.Request(url, method="GET")
    req.add_header("User-Agent", "Mozilla/5.0 (compatible; Mermaid-Renderer/1.0)")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec
        return resp.read(), resp.headers.get("Content-Type", "")

def _http_post(url: str, data: bytes, content_type: str, timeout: int = 30) -> Tuple[bytes, str]:
    import urllib.request
    req = urllib.request.Request(url, data=data, headers={
        "Content-Type": content_type,
        "User-Agent": "Mozilla/5.0 (compatible; Mermaid-Renderer/1.0)"
    }, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec
        return resp.read(), resp.headers.get("Content-Type", "")

def _is_png(bytes_: bytes, content_type: str) -> bool:
    return bytes_.startswith(b"\x89PNG\r\n\x1a\n") or ("image/png" in (content_type or "").lower())

def _encode_mermaid_ink(code: str) -> str:
    """Fixed Mermaid.ink encoder with proper URL encoding"""
    try:
        # Clean the mermaid code - remove any extra whitespace
        cleaned_code = code.strip()
        
        # Use standard zlib compression
        compressed = zlib.compress(cleaned_code.encode("utf-8"))
        
        # Use standard base64 encoding (not URL-safe initially)
        b64 = base64.b64encode(compressed).decode("ascii")
        
        # Now make it URL-safe
        b64_urlsafe = b64.replace("+", "-").replace("/", "_").rstrip("=")
        
        return f"https://mermaid.ink/img/{b64_urlsafe}"
    except Exception as e:
        raise RuntimeError(f"Failed to encode for mermaid.ink: {e}")

def _render_png_via_mermaid_ink(diagram: str) -> Tuple[bytes, str]:
    """Improved Mermaid.ink renderer with better error handling"""
    try:
        url = _encode_mermaid_ink(diagram)
        print(f"Requesting: {url[:100]}...")  # Debug output
        data, ctype = _http_get(url, timeout=45)
        if _is_png(data, ctype):
            return data, "mermaid.ink"
        # Check if it's an error response
        if data.startswith(b"<!DOCTYPE") or b"error" in data.lower():
            error_text = data[:200].decode('utf-8', errors='ignore')
            raise RuntimeError(f"Mermaid.ink returned error page: {error_text}")
        raise RuntimeError(f"Mermaid.ink returned non-PNG (content-type={ctype!r}, size={len(data)}).")
    except Exception as e:
        raise RuntimeError(f"Mermaid.ink rendering failed: {e}")

def _render_png_via_kroki(diagram: str) -> Tuple[bytes, str]:
    """Improved Kroki renderer"""
    try:
        # Try text/plain first (simpler)
        print("Trying Kroki text/plain method...")
        data, ctype = _http_post("https://kroki.io/mermaid/png", diagram.encode("utf-8"), "text/plain; charset=utf-8", 45)
        if _is_png(data, ctype):
            return data, "kroki.io(text)"
        
        # Try JSON method
        print("Trying Kroki JSON method...")
        body = json.dumps({"diagram_source": diagram, "diagram_type": "mermaid", "output_format": "png"}).encode("utf-8")
        data, ctype = _http_post("https://kroki.io/mermaid/png", body, "application/json; charset=utf-8", 45)
        if _is_png(data, ctype):
            return data, "kroki.io(json)"
            
        raise RuntimeError(f"Kroki returned non-PNG (content-type={ctype!r}, size={len(data)}).")
    except Exception as e:
        raise RuntimeError(f"Kroki rendering failed: {e}")

# ---------------- Local renderer (mmdc) ----------------

def _have_mmdc() -> Optional[str]:
    # Prefer explicit env var first
    explicit = os.getenv("MERMAID_CLI")  # full path to mmdc
    if explicit and Path(explicit).exists():
        return explicit
    which = shutil.which("mmdc")
    if which:
        return which
    # Allow npx usage if desired (requires internet on first run)
    npx = shutil.which("npx")
    if npx:
        return "npx -y @mermaid-js/mermaid-cli mmdc"  # used via shell=True
    return None

def _run_mmdc_to_png(diagram: str, out_path: Path) -> str:
    mmdc = _have_mmdc()
    if not mmdc:
        raise FileNotFoundError(
            "Mermaid CLI (mmdc) not found. Install with `npm i -g @mermaid-js/mermaid-cli` "
            "or set MERMAID_CLI to its full path."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        inp = Path(td) / "diagram.mmd"
        inp.write_text(diagram, encoding="utf-8")

        # Optional Chromium path for corp/offline boxes
        chrome = os.getenv("MERMAID_CHROMIUM_PATH") or os.getenv("PUPPETEER_EXECUTABLE_PATH")
        puppeteer_cfg = None
        cmd: list[str] | str

        if chrome and Path(chrome).exists():
            puppeteer_cfg = Path(td) / "puppeteer.json"
            puppeteer_cfg.write_text(json.dumps({"executablePath": str(Path(chrome).resolve())}), encoding="utf-8")

        if " " in (mmdc or "") and mmdc.startswith("npx "):
            # Use shell to support inline "npx -y ..."
            cmd = f'{mmdc} -i "{inp}" -o "{out_path}"{" -p " + str(puppeteer_cfg) if puppeteer_cfg else ""}'
            proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        else:
            cmd_list = [mmdc, "-i", str(inp), "-o", str(out_path)]
            if puppeteer_cfg:
                cmd_list += ["-p", str(puppeteer_cfg)]
            proc = subprocess.run(cmd_list, shell=False, capture_output=True, text=True)

        if proc.returncode != 0:
            raise RuntimeError(f"mmdc failed (code={proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}")

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError("mmdc reported success but PNG not found or empty.")
    return f"Saved Mermaid PNG to {out_path} ({out_path.stat().st_size} bytes) via mmdc."


def _validate_mermaid_syntax(diagram: str) -> str:
    """Validate and clean Mermaid syntax"""
    lines = [line.strip() for line in diagram.split('\n') if line.strip()]
    
    # Basic validation - ensure it starts with a valid diagram type
    if not lines:
        raise ValueError("Empty diagram")
        
    first_line = lines[0].lower()
    valid_starts = ['graph', 'flowchart', 'sequencediagram', 'classdiagram', 'statediagram', 'erdiagram', 'gantt']
    
    if not any(first_line.startswith(start) for start in valid_starts):
        # If it doesn't start with a valid diagram type, assume it's a flowchart
        lines.insert(0, "flowchart TD")
    
    return '\n'.join(lines)


def make_mermaid_tools(project_dir: str):
    @tool("mermaid_draw", return_direct=False)
    def mermaid_draw(
        nodes: Any,
        edges: Any,
        first_node: Optional[str] = None,
        last_node: Optional[str] = None,
        with_styles: bool = True,
        curve_style: str = "LINEAR",
        node_styles: Any = None,
        wrap_label_n_words: int = 9,
        frontmatter_config: Any = None,
        fenced: bool = True,
    ) -> str:
        """
        Build valid Mermaid syntax from a graph definition.

        Use this tool when:
        - You need to generate Mermaid code from JSON/YAML structures.
        - You want to normalize or validate raw Mermaid syntax.
        - You need to insert start/end nodes or apply styles.

        Accepted input formats:
        - **Raw Mermaid** (string starting with `graph`, `flowchart`, etc.)
        - **Nodes as dict/list**:
            {"A": "Start", "B": "End"}
        - **Edges as list/dict**:
            [{"source": "A", "target": "B"}]
        - YAML equivalents of the above.

        Syntax rules (same as `mermaid_png`):
        - Use only valid Mermaid node shapes:
            A[Process], A(Rounded), A{Decision}, A([Stadium])
        - Edge labels: `A -->|yes| B` (no quotes `"yes"`)
        - Keep labels concise, no raw code inside nodes.

        Output:
        - By default, returns a fenced block:
                ```mermaid
                graph TD
                A[Start] --> B[End]
                ```
        - If `fenced=False`, returns only the Mermaid text body.
        """
        if not _MERMAID_OK:
            return (
                "Mermaid support unavailable. "
                "Please upgrade `langchain-core` to a version that includes `graph_mermaid`."
            )

        # If caller passed YAML/JSON-like data for node_styles, ignore it.
        # We only forward actual NodeStyles instances to draw_mermaid.
        if isinstance(node_styles, (str, dict, list, tuple)):
            node_styles = None

        mermaid_nodes = _looks_like_mermaid(nodes)
        mermaid_edges = _looks_like_mermaid(edges)
        if mermaid_nodes or mermaid_edges:
            syntax = mermaid_nodes or mermaid_edges or ""
            validated = _validate_mermaid_syntax(syntax)
            return f"```mermaid\n{validated}\n```" if fenced else validated

        try:
            nodes_obj_or_str = _norm_nodes(nodes)
            if isinstance(nodes_obj_or_str, str):
                syntax = nodes_obj_or_str
                validated = _validate_mermaid_syntax(syntax)
                return f"```mermaid\n{validated}\n```" if fenced else validated
            nodes_obj: dict[str, str] = nodes_obj_or_str

            edges_list, inferred_from_edges, mermaid_from_edges = _norm_edges(edges)
            if mermaid_from_edges:
                validated = _validate_mermaid_syntax(mermaid_from_edges)
                return f"```mermaid\n{validated}\n```" if fenced else validated

            for nid in inferred_from_edges.keys():
                nodes_obj.setdefault(nid, nid)

            if first_node and first_node not in nodes_obj:
                nodes_obj[str(first_node)] = str(first_node)
            if last_node and last_node not in nodes_obj:
                nodes_obj[str(last_node)] = str(last_node)

            # Only frontmatter supports JSON/YAML. Node styles must be a real NodeStyles object.
            fm_obj = _loads_json_or_yaml(frontmatter_config, None)

            try:
                cs = CurveStyle[curve_style.upper().strip()]
            except Exception:
                cs = CurveStyle.LINEAR

            # Safely determine NodeStyles instance (or None)
            ns = None
            if _HAS_NODESTYLES and node_styles is not None:
                try:
                    from langchain_core.runnables.graph_mermaid import NodeStyles as _NodeStyles  # type: ignore
                    if isinstance(node_styles, _NodeStyles):
                        ns = node_styles
                except Exception:
                    ns = None

            try:
                syntax = draw_mermaid(
                    nodes=nodes_obj,
                    edges=edges_list,
                    first_node=first_node,
                    last_node=last_node,
                    with_styles=with_styles,
                    curve_style=cs,
                    node_styles=ns,  # only a proper NodeStyles or None ever gets through
                    wrap_label_n_words=wrap_label_n_words,
                    frontmatter_config=fm_obj,
                )
            except AttributeError:
                # Some langchain-core versions may attempt `.name` on enums/styles;
                # retry with node_styles disabled.
                syntax = draw_mermaid(
                    nodes=nodes_obj,
                    edges=edges_list,
                    first_node=first_node,
                    last_node=last_node,
                    with_styles=with_styles,
                    curve_style=cs,
                    node_styles=None,
                    wrap_label_n_words=wrap_label_n_words,
                    frontmatter_config=fm_obj,
                )
            except TypeError:
                # Older signatures without node_styles parameter.
                syntax = draw_mermaid(
                    nodes=nodes_obj,
                    edges=edges_list,
                    first_node=first_node,
                    last_node=last_node,
                    with_styles=with_styles,
                    curve_style=cs,
                    node_styles=None,
                    wrap_label_n_words=wrap_label_n_words,
                    frontmatter_config=fm_obj,
                )

            validated = _validate_mermaid_syntax(syntax)
            return f"```mermaid\n{validated}\n```" if fenced else validated

        except Exception as e:
            return f"Error generating Mermaid: {type(e).__name__}: {e}"

    @tool("mermaid_png", return_direct=False)
    def mermaid_png(
        mermaid_syntax: str,
        output_file_path: Optional[str] = None,
        draw_method: str = "AUTO",   # AUTO | API | LOCAL (kept for backward compat)
        background_color: Optional[str] = "white",
        padding: int = 10,
        max_retries: int = 2,  # Increased retries
        retry_delay: float = 1.0,
    ) -> str:
        """
        Render Mermaid syntax to a PNG file inside the repo and return the saved path.
        Accepts fenced blocks; extracts inner Mermaid automatically.

        IMPORTANT SYNTAX RULES (to avoid mmdc/Kroki parse errors):
        - Always use **valid Mermaid node shapes**:
            - Rectangle (process):   A[Do something]
            - Round edges:           A(Rounded step)
            - Diamond (decision):    A{Yes or No?}
            - Stadium:               A([Stadium step])
        - **Do not** use parentheses or braces with raw text like: A{foo(bar)} or A(classify_complexity(query)).
            Instead, write clean labels like: A{Classify complexity?}
        - **Do not** wrap edge labels in quotes → use `A -->|simple| B` not `A --> "simple" --> B`.
        - Keep labels short, no raw punctuation-heavy code inside nodes. For code, rephrase into natural language.

        Modes:
        - env MERMAID_RENDER_MODE=local|api|auto (default auto)
        - draw_method overrides per-call (LOCAL/API/AUTO)

        Rendering priority:
        1. LOCAL (via Mermaid CLI `mmdc`)
        2. API fallback (Mermaid.ink, Kroki, etc.)
        
        Returns:
        A message with the relative PNG path + size and source used.
        """
        if not _MERMAID_OK:
            return (
                "Mermaid rendering unavailable. "
                "Please upgrade `langchain-core` to include `graph_mermaid`."
            )

        body = mermaid_syntax or ""
        m = re.search(r"```(?:mermaid)?\s*(.*?)```", body, flags=re.S | re.I)
        if m:
            body = m.group(1).strip()
        if not body.strip():
            return "Error rendering Mermaid PNG: empty mermaid_syntax."

        # Validate the mermaid syntax
        try:
            body = _validate_mermaid_syntax(body)
            print(f"Validated Mermaid syntax:\n{body}")
        except ValueError as e:
            return f"Error: Invalid Mermaid syntax: {e}"

        # Ensure .png filename
        rel = output_file_path or f"assets/mermaid_{int(time.time())}.png"
        if not str(rel).lower().endswith(".png"):
            rel = f"{rel}.png"

        try:
            out_path = _rooted(project_dir, str(rel))
        except Exception as e:
            return f"Invalid output path: {e}"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Decide mode
        env_mode = (os.getenv("MERMAID_RENDER_MODE") or "AUTO").strip().upper()
        call_mode = (draw_method or "AUTO").strip().upper()
        mode = call_mode if call_mode in {"LOCAL", "API", "AUTO"} else env_mode
        if mode not in {"LOCAL", "API", "AUTO"}:
            mode = "AUTO"

        # Helper to try saving bytes
        def _save_and_report(png: bytes, source: str) -> str:
            out_path.write_bytes(png)
            return f"Saved Mermaid PNG to {out_path.relative_to(Path(project_dir))} ({len(png)} bytes) via {source}."

        last_error = None
        
        # Try LOCAL first (AUTO path prefers local for offline reliability)
        if mode in {"LOCAL", "AUTO"}:
            try:
                return _run_mmdc_to_png(body, out_path)
            except Exception as e_local:
                last_error = f"Local: {e_local}"
                print(f"Local rendering failed: {e_local}")
                if mode == "LOCAL":
                    return f"Error rendering Mermaid PNG locally: {type(e_local).__name__}: {e_local}"

        # Try API(s) with retries
        for attempt in range(max_retries):
            try:
                print(f"Trying Mermaid.ink (attempt {attempt + 1}/{max_retries})")
                png, source = _render_png_via_mermaid_ink(body)
                return _save_and_report(png, source)
            except Exception as e_mermaid:
                last_error = f"Mermaid.ink: {e_mermaid}"
                print(f"Mermaid.ink failed: {e_mermaid}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)

        # Try Kroki as final fallback
        try:
            print("Trying Kroki as final fallback...")
            png, source = _render_png_via_kroki(body)
            return _save_and_report(png, source)
        except Exception as e_kroki:
            last_error = f"Kroki: {e_kroki}"
            print(f"Kroki failed: {e_kroki}")

        return f"Error rendering Mermaid PNG: All methods failed. Last errors: {last_error}"

    return [mermaid_draw, mermaid_png]
