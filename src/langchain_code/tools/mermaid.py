from __future__ import annotations
from pathlib import Path
from typing import Any, Optional, Iterable
import json
import time
import re

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
    """Accept dict/list already, or parse JSON/YAML strings. Fallback to default."""
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
    """If `src` is a string that already looks like Mermaid syntax, return it."""
    if not isinstance(src, str):
        return None
    s = src.strip()
    if re.search(r"\b(graph|flowchart|sequenceDiagram|classDiagram|stateDiagram|erDiagram|gantt)\b", s):
        m = re.search(r"```(?:mermaid)?\s*(.*?)```", s, flags=re.S | re.I)
        return m.group(1).strip() if m else s
    return None


def _norm_nodes(raw: Any) -> dict[str, str] | str:
    """Normalize node formats into {id: label}. If Mermaid string, return it."""
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
    """Parse string edges like 'A->B: label' or 'A-->B|label|' into normalized dict."""
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
    """
    Normalize edges into list of dicts; also return nodes inferred from edges.
    Returns (edges, inferred_nodes, mermaid_passthrough).
    """
    m = _looks_like_mermaid(raw)
    if m is not None:
        return ([], {}, m)

    raw = _loads_json_or_yaml(raw, [])
    edges: list[dict] = []
    inferred_nodes: dict[str, str] = {}

    if isinstance(raw, dict):
        # adjacency: {"A": ["B","C"]} or {"A":[{"to":"B","label":"x"}]}
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


# --- HTTP fallback (API-only) -------------------------------------------------

def _http_post_bytes(url: str, data: bytes, content_type: str = "text/plain; charset=utf-8", timeout: int = 20) -> bytes:
    """
    Minimal HTTP POST using stdlib only (no requests).
    """
    import urllib.request
    req = urllib.request.Request(url, data=data, headers={"Content-Type": content_type}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec - user opted into API rendering
        return resp.read()


def _render_png_via_api_fallback(diagram: str) -> tuple[bytes, str]:
    """
    Try Mermaid.ink first, then Kroki. Returns (png_bytes, source_name).
    """
    payload = diagram.encode("utf-8")

    # 1) Mermaid.ink POST
    try:
        png = _http_post_bytes("https://mermaid.ink/img/", payload, "text/plain; charset=utf-8", timeout=30)
        if png and len(png) > 0:
            return png, "mermaid.ink"
    except Exception:
        pass

    # 2) Kroki (text/plain endpoint)
    try:
        png = _http_post_bytes("https://kroki.io/mermaid/png", payload, "text/plain; charset=utf-8", timeout=30)
        if png and len(png) > 0:
            return png, "kroki.io"
    except Exception:
        pass

    # 3) Kroki JSON endpoint (just in case)
    try:
        body = json.dumps({"diagram_source": diagram}).encode("utf-8")
        png = _http_post_bytes("https://kroki.io/mermaid/png", body, "application/json; charset=utf-8", timeout=30)
        if png and len(png) > 0:
            return png, "kroki.io(json)"
    except Exception:
        pass

    raise RuntimeError("All API renderers failed (mermaid.ink, kroki.io).")


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
        Build Mermaid syntax from a graph (nodes/edges) and return it.

        Accepted node formats:
        - {"id": "Label", ...}
        - [{"id":"A","label":"Alpha"}, {"id":"B","label":"Beta"}]
        - ["A","B","C"]  (label=id)
        - Raw Mermaid string (graph/flowchart/etc.) -> passed through

        Accepted edge formats:
        - [{"source":"A","target":"B","data":{"label":"..."}}, ...]
        - [{"from":"A","to":"B","label":"..."}]
        - [["A","B","...label..."], ...]
        - {"A":["B","C"], "D":[{"to":"E","label":"..."}]}  (adjacency)
        - ["A->B: label", "X-->Y|label|"]  (strings)
        - Raw Mermaid string -> passed through

        All parameters also accept JSON (or YAML) strings.
        """
        if not _MERMAID_OK:
            return (
                "Mermaid support unavailable. "
                "Please upgrade `langchain-core` to a version that includes `graph_mermaid`."
            )

        # Passthrough if either is already Mermaid
        mermaid_nodes = _looks_like_mermaid(nodes)
        mermaid_edges = _looks_like_mermaid(edges)
        if mermaid_nodes or mermaid_edges:
            syntax = mermaid_nodes or mermaid_edges or ""
            return f"```mermaid\n{syntax}\n```" if fenced else syntax

        try:
            nodes_obj_or_str = _norm_nodes(nodes)
            if isinstance(nodes_obj_or_str, str):
                syntax = nodes_obj_or_str
                return f"```mermaid\n{syntax}\n```" if fenced else syntax
            nodes_obj: dict[str, str] = nodes_obj_or_str

            edges_list, inferred_from_edges, mermaid_from_edges = _norm_edges(edges)
            if mermaid_from_edges:
                return f"```mermaid\n{mermaid_from_edges}\n```" if fenced else mermaid_from_edges

            # Ensure all edge nodes exist
            for nid in inferred_from_edges.keys():
                nodes_obj.setdefault(nid, nid)

            # Respect explicit endpoints
            if first_node and first_node not in nodes_obj:
                nodes_obj[str(first_node)] = str(first_node)
            if last_node and last_node not in nodes_obj:
                nodes_obj[str(last_node)] = str(last_node)

            node_styles_obj = _loads_json_or_yaml(node_styles, None)
            fm_obj = _loads_json_or_yaml(frontmatter_config, None)

            try:
                cs = CurveStyle[curve_style.upper().strip()]
            except Exception:
                cs = CurveStyle.LINEAR

            try:
                syntax = draw_mermaid(
                    nodes=nodes_obj,
                    edges=edges_list,
                    first_node=first_node,
                    last_node=last_node,
                    with_styles=with_styles,
                    curve_style=cs,
                    node_styles=node_styles_obj if _HAS_NODESTYLES else None,
                    wrap_label_n_words=wrap_label_n_words,
                    frontmatter_config=fm_obj,
                )
            except TypeError:
                # Retry without node_styles for older LC versions
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

            return f"```mermaid\n{syntax}\n```" if fenced else syntax

        except Exception as e:
            return f"Error generating Mermaid: {type(e).__name__}: {e}"

    @tool("mermaid_png", return_direct=False)
    def mermaid_png(
        mermaid_syntax: str,
        output_file_path: Optional[str] = None,
        draw_method: str = "API",  # ignored; we force API-only below
        background_color: Optional[str] = "white",
        padding: int = 10,
        max_retries: int = 1,
        retry_delay: float = 1.0,
    ) -> str:
        """
        Render Mermaid syntax to a PNG file inside the repo and return the saved path.
        Accepts fenced blocks; extracts inner Mermaid automatically.

        This implementation **forces API rendering only** and never requires pyppeteer.
        """
        if not _MERMAID_OK:
            return (
                "Mermaid rendering unavailable. "
                "Please upgrade `langchain-core` to include `graph_mermaid`."
            )

        # Extract body if fenced
        body = mermaid_syntax or ""
        m = re.search(r"```(?:mermaid)?\s*(.*?)```", body, flags=re.S | re.I)
        if m:
            body = m.group(1).strip()
        if not body.strip():
            return "Error rendering Mermaid PNG: empty mermaid_syntax."

        # Ensure .png filename
        rel = output_file_path or f"assets/mermaid_{int(time.time())}.png"
        if not str(rel).lower().endswith(".png"):
            rel = f"{rel}.png"

        try:
            out_path = _rooted(project_dir, str(rel))
        except Exception as e:
            return f"Invalid output path: {e}"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # 1) Try LangChain's API renderer explicitly (force API enum)
        try:
            method = MermaidDrawMethod.API  # force API-only
            data = draw_mermaid_png(
                mermaid_syntax=body,
                output_file_path=str(out_path),
                draw_method=method,
                background_color=background_color,
                padding=padding,
                max_retries=max_retries,
                retry_delay=retry_delay,
            )
            if not out_path.exists() and data:
                out_path.write_bytes(data)
            if out_path.exists():
                return f"Saved Mermaid PNG to {out_path.relative_to(Path(project_dir))} ({out_path.stat().st_size} bytes)."
        except Exception as e:
            # If the internal path tries to use pyppeteer or anything else, fall through to raw HTTP APIs.
            pass

        # 2) Fallback: direct HTTP renderers (mermaid.ink → kroki.io)
        try:
            png, source = _render_png_via_api_fallback(body)
            out_path.write_bytes(png)
            return f"Saved Mermaid PNG to {out_path.relative_to(Path(project_dir))} ({len(png)} bytes) via {source}."
        except Exception as e:
            return f"Error rendering Mermaid PNG: {type(e).__name__}: {e}"

    return [mermaid_draw, mermaid_png]
