#!/usr/bin/env python3
"""Render Mermaid flowcharts to laid-out SVG, from the Markdown that holds them.

Mermaid is the source of truth for a diagram's **structure**: nodes,
groups (``subgraph``), edges with their labels, classes and styles. It
renders anywhere (GitHub, PyCharm) with an automatic layout that cannot
draw a dense, nested picture well. This script draws such a picture from
the **same block**: its layout hints live in Mermaid comments, which every
Mermaid renderer ignores, so the Markdown stays the single source.

A block is rendered only when it carries an ``@out`` hint. Hints, one per
comment line (``%% @...``); coordinates are SVG pixels:

- ``@out PATH`` — the SVG to write, relative to the Markdown file; it must
  be an ``.svg`` file under the Markdown file's directory (symlinks
  resolved), so ``render`` never writes anywhere else.
- ``@canvas W H [origin=X,Y]`` — the drawing size and its top-left corner.
- ``@group ID at=X,Y size=W,H [title=top|bottom] [badge=TEXT]``
- ``@node ID at=X,Y size=W,H [fs=N] [tfs=N]`` — every node needs one.
- ``@chips ID NAME=COLOR ...`` — a ``[[queue]]`` node's last label line,
  split on ``·``, drawn as coloured chips.
- ``@edge A->B`` (or ``A<->B``, as written in the block) with a route:
  ``vhv x1= y= x2=`` (vertical, horizontal, vertical), ``hv y1= x2=``,
  ``hvh y1= x= y2=``, or a straight ``x=`` / ``y=``; plus ``color=NAME``,
  ``label=X,Y[,anchor]`` or ``label=none``, ``wrap=N``, ``fs=N``. Without
  a route an edge is straight when the two boxes overlap on one axis, and
  a vertical Z otherwise.
- ``@text X,Y "free text"``

Supported Mermaid subset: ``flowchart``/``graph``/``direction`` lines,
nodes ``id["…"]`` (box), ``id(["…"])`` (stadium), ``id[("…")]``
(cylinder), ``id[["…"]]`` (queue), ``id{"…"}`` (diamond), ``subgraph ID["…"]`` … ``end``, edges ``-->``,
``-.->``, ``---``, ``-.-`` (optionally prefixed by ``<``) with an optional
``|"label"|``, ``classDef``, ``class``, ``style``. Labels split on
``<br/>``; ``<b>`` is dropped. A node title ending in ``· LLM`` gets an
``LLM`` badge instead. Anything else is refused, never guessed.

Usage::

    python scripts/qs/mermaid_svg.py render docs/epics/QS-369.md
    python scripts/qs/mermaid_svg.py render --check docs/epics/QS-369.md

Contract: JSON on stdout; exit 0 when every SVG was written (``render``)
or is up to date (``--check``), exit 1 on a stale SVG or a refused block.
No side effects at import.

Two other callers (QS-404):

- ``quality_gate.py --impacted`` imports this module and runs
  :func:`run` with ``check=True`` over the changed ``docs/`` Markdown,
  so a stale SVG fails before commit as it does in CI.
- ``epic_doc.py land`` reads **origin/main's** copy of this file with
  ``git show`` and execs it into a fresh module, then calls
  :func:`outputs_from_text` on the epic document it lands, so the SVGs
  it lands are the ones CI on ``main`` expects.

Import contract (relied on by ``epic_doc.py land``, pinned by a test):
stdlib-only imports; no use of the module's file, spec or any data file;
no side effect at import — module level only imports, defines and
assigns.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shlex
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

COLORS = {
    "orange": "#E8590C",
    "indigo": "#3B5BDB",
    "blue": "#3B5BDB",
    "teal": "#0C8599",
    "green": "#2B8A3E",
    "purple": "#6741D9",
    "gray": "#495057",
    "red": "#C92A2A",
    "black": "#343A40",
}
INK = "#212529"
GRAY = "#495057"
NEUTRAL_STROKES = {"#343A40", "#868E96"}
FONT = "Inter, -apple-system, Segoe UI, Helvetica, Arial, sans-serif"


class MermaidSvgError(ValueError):
    """A block this script refuses to render (unsupported line, missing hint)."""


@dataclass
class Node:
    id: str
    shape: str
    lines: list[str]
    cls: str | None = None
    style: dict[str, str] = field(default_factory=dict)


@dataclass
class Group:
    id: str
    title: str
    style: dict[str, str] = field(default_factory=dict)


@dataclass
class Edge:
    a: str
    b: str
    both: bool
    dashed: bool
    arrow: bool
    label: str | None

    @property
    def key(self) -> str:
        return f"{self.a}{'<->' if self.both else '->'}{self.b}"


@dataclass
class Graph:
    nodes: dict[str, Node] = field(default_factory=dict)
    groups: dict[str, Group] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    classdefs: dict[str, dict[str, str]] = field(default_factory=dict)
    hints: list[str] = field(default_factory=list)


@dataclass
class Hints:
    out: str | None = None
    canvas: tuple[float, float, float, float] | None = None
    group: dict[str, dict[str, str]] = field(default_factory=dict)
    node: dict[str, dict[str, str]] = field(default_factory=dict)
    chips: dict[str, dict[str, str]] = field(default_factory=dict)
    edge: dict[str, dict[str, str]] = field(default_factory=dict)
    text: list[tuple[float, float, str]] = field(default_factory=list)


# --------------------------------------------------------------------- parsing
NODE_RE = re.compile(r'^(\w+)(\[\(|\[\[|\(\[|\[|\{)"(.*)"(\)\]|\]\]|\]\)|\]|\})$')
EDGE_STEP_RE = re.compile(r'\s*(<?)(-->|-\.->|-\.-|---)\s*(?:\|"?(.*?)"?\|)?\s*(\w+)')
SUBGRAPH_RE = re.compile(r'^subgraph\s+(\w+)\["(.*)"\]$')
SHAPES = {"[(": "cyl", "[[": "queue", "([": "stadium", "[": "box", "{": "diamond"}


def label_lines(label: str) -> list[str]:
    """Split a Mermaid label on ``<br/>`` and drop ``<b>`` tags."""
    label = re.sub(r"</?b>", "", label)
    return [part.strip() for part in re.split(r"<br\s*/?>", label)]


def parse_props(text: str) -> dict[str, str]:
    """Parse ``fill:#fff,stroke:#000`` into a dict."""
    out = {}
    for part in text.split(","):
        key, sep, value = part.partition(":")
        if sep:
            out[key.strip()] = value.strip()
    return out


def parse(block: str) -> Graph:
    """Parse one Mermaid flowchart block (the supported subset)."""
    graph = Graph()
    stack: list[str] = []
    for raw in block.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("%%"):
            body = line[2:].strip()
            if body.startswith("@"):
                graph.hints.append(body[1:])
            continue
        if line.startswith(("flowchart", "graph", "direction")):
            continue
        if match := SUBGRAPH_RE.match(line):
            graph.groups[match.group(1)] = Group(match.group(1), label_lines(match.group(2))[0])
            stack.append(match.group(1))
            continue
        if line == "end":
            if not stack:
                raise MermaidSvgError(f"'end' without a subgraph: {raw!r}")
            stack.pop()
            continue
        if line.startswith("classDef "):
            name, _, props = line[len("classDef ") :].partition(" ")
            graph.classdefs[name] = parse_props(props)
            continue
        if line.startswith("class "):
            ids, _, name = line[len("class ") :].rpartition(" ")
            for ident in ids.split(","):
                _lookup_node(graph, ident.strip(), raw).cls = name
            continue
        if line.startswith("style "):
            ident, _, props = line[len("style ") :].partition(" ")
            target = graph.groups.get(ident) or _lookup_node(graph, ident, raw)
            target.style = parse_props(props)
            continue
        if match := NODE_RE.match(line):
            ident = match.group(1)
            graph.nodes[ident] = Node(ident, SHAPES[match.group(2)], label_lines(match.group(3)))
            continue
        if (edges := parse_edges(line)) is not None:
            graph.edges.extend(edges)
            continue
        raise MermaidSvgError(f"unsupported Mermaid line: {raw!r}")
    if stack:
        raise MermaidSvgError(f"subgraph {stack[-1]!r} is never closed")
    return graph


def parse_edges(line: str) -> list[Edge] | None:
    """The edges of a line like ``a -->|"x"| b -.-> c``, or None if it is not one."""
    first = re.match(r"\w+", line)
    if first is None:
        return None
    edges, a, pos = [], first.group(0), first.end()
    while pos < len(line):
        step = EDGE_STEP_RE.match(line, pos)
        if step is None:
            return None
        back, kind, label, b = step.groups()
        edges.append(Edge(a, b, bool(back), "." in kind, kind in ("-->", "-.->"), label))
        a, pos = b, step.end()
    return edges or None


def _lookup_node(graph: Graph, ident: str, raw: str) -> Node:
    if ident not in graph.nodes:
        raise MermaidSvgError(f"unknown node {ident!r} in: {raw!r}")
    return graph.nodes[ident]


def _pair(value: str) -> tuple[float, float]:
    x, _, y = value.partition(",")
    return float(x), float(y)


def parse_hints(lines: list[str]) -> Hints:
    """Parse the ``@...`` comment lines of a block."""
    hints = Hints()
    for line in lines:
        parts = shlex.split(line)
        kind, rest = parts[0], parts[1:]
        if kind == "out":
            hints.out = rest[0]
        elif kind == "canvas":
            options = dict(part.partition("=")[::2] for part in rest[2:])
            ox, oy = _pair(options.get("origin", "0,0"))
            hints.canvas = (float(rest[0]), float(rest[1]), ox, oy)
        elif kind == "text":
            x, y = _pair(rest[0])
            hints.text.append((x, y, rest[1]))
        elif kind in ("group", "node", "chips", "edge"):
            options = {}
            for part in rest[1:]:
                key, sep, value = part.partition("=")
                options[key] = value if sep else "yes"
            getattr(hints, kind)[rest[0]] = options
        else:
            raise MermaidSvgError(f"unknown layout hint: @{line}")
    return hints


# --------------------------------------------------------------------- drawing
def esc(value: str) -> str:
    return html.escape(value, quote=True)


def svg_text(
    x: float,
    y: float,
    lines: list[str],
    size: float = 12,
    color: str = INK,
    anchor: str = "start",
    weight: str = "normal",
    halo: bool = False,
    line_height: float | None = None,
) -> str:
    """A multi-line ``<text>``; ``halo`` draws a white outline behind it."""
    line_height = line_height or size * 1.28
    attrs = f'font-size="{size:g}" fill="{color}" text-anchor="{anchor}" font-weight="{weight}"'
    if halo:
        attrs += ' stroke="#fff" stroke-width="4" paint-order="stroke" stroke-linejoin="round"'
    spans = "".join(
        f'<tspan x="{x:g}" dy="{0 if i == 0 else line_height:g}">{esc(line)}</tspan>' for i, line in enumerate(lines)
    )
    return f'<text x="{x:g}" y="{y:g}" {attrs}>{spans}</text>'


def wrap(value: str, width: float, size: float) -> list[str]:
    """Wrap to the characters that fit ``width`` pixels at font ``size``."""
    return textwrap.wrap(value, max(8, int(width / (size * 0.55)))) or [""]


def badge(x: float, y: float, label: str) -> str:
    color, fill = ("#C2255C", "#FFF0F6") if label == "LLM" else ("#343A40", "#E9ECEF")
    return (
        f'<rect x="{x:g}" y="{y:g}" width="40" height="18" rx="4" fill="{fill}" stroke="{color}" stroke-width="1.2"/>'
        + svg_text(x + 20, y + 13, [label], 10.5, color, "middle", "bold")
    )


Box = tuple[float, float, float, float]


class Renderer:
    """Draw a parsed graph with its hints. Every node and group needs a hint."""

    def __init__(self, graph: Graph, hints: Hints) -> None:
        if hints.canvas is None:
            raise MermaidSvgError("missing @canvas hint")
        self.graph = graph
        self.hints = hints
        self.parts: list[str] = []
        self.colors: set[str] = set()
        self.boxes: dict[str, Box] = {}

    def _placement(self, kind: str, ident: str) -> tuple[float, float, float, float]:
        options = getattr(self.hints, kind).get(ident)
        if options is None or "at" not in options or "size" not in options:
            raise MermaidSvgError(f"missing @{kind} hint (at=, size=) for {ident!r}")
        x, y = _pair(options["at"])
        w, h = _pair(options["size"])
        return x, y, w, h

    def _node_style(self, node: Node) -> dict[str, str]:
        return {**self.graph.classdefs.get(node.cls or "", {}), **node.style}

    def edge_color(self, ident: str) -> str:
        style = (
            self.graph.groups[ident].style if ident in self.graph.groups else self._node_style(self.graph.nodes[ident])
        )
        stroke = style.get("stroke", GRAY)
        return GRAY if stroke.upper() in NEUTRAL_STROKES else stroke

    def draw_group(self, group: Group) -> None:
        x, y, w, h = self._placement("group", group.id)
        options = self.hints.group[group.id]
        self.boxes[group.id] = (x, y, w, h)
        stroke = group.style.get("stroke", GRAY)
        fill = group.style.get("fill", "#F8F9FA")
        dash = ' stroke-dasharray="6 4"' if "stroke-dasharray" in group.style else ""
        self.parts.append(
            f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="2.5"{dash}/>'
        )
        bottom = options.get("title") == "bottom"
        title_x, title_y = (x + 16, y + h - 9) if bottom else (x + 14, y + 22)
        self.parts.append(svg_text(title_x, title_y, [group.title], 15, stroke, weight="bold"))
        if "badge" in options:
            self.parts.append(badge(x + w - 52, y + h - 24 if bottom else y + 8, options["badge"]))

    def draw_node(self, node: Node) -> None:
        x, y, w, h = self._placement("node", node.id)
        options = self.hints.node[node.id]
        size = float(options.get("fs", 12))
        title_size = float(options.get("tfs", 13.5))
        style = self._node_style(node)
        stroke, fill = style.get("stroke", GRAY), style.get("fill", "#fff")
        title, body = node.lines[0], node.lines[1:]
        llm = title.endswith("· LLM")
        if llm:
            title = title[: -len("· LLM")].rstrip()
        if node.shape == "cyl":
            self.boxes[node.id] = (x, y - 14, w, h + 28)
            self._draw_cylinder(x, y, w, h, stroke, fill, title, body, title_size)
            return
        self.boxes[node.id] = (x, y, w, h)
        if node.shape == "diamond":
            points = f"{x + w / 2:g},{y:g} {x + w:g},{y + h / 2:g} {x + w / 2:g},{y + h:g} {x:g},{y + h / 2:g}"
            self.parts.append(f'<polygon points="{points}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
            self.parts.append(svg_text(x + w / 2, y + h / 2 + 4, [title, *body], size, INK, "middle"))
            return
        queue = node.shape == "queue"
        radius = h / 2 if node.shape == "stadium" else 8 if queue else 10
        dash = ' stroke-dasharray="6 4"' if "stroke-dasharray" in style else ""
        self.parts.append(
            f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="{radius:g}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{2.5 if queue else 2}"{dash}/>'
        )
        if queue and node.id in self.hints.chips:
            self._draw_chips(node, x, y, w, stroke, title, body)
            return
        pad = 10 + (h / 4 if node.shape == "stadium" else 0)  # clear a stadium's rounded ends
        cursor = y + title_size + 8
        for line in wrap(title, w - 2 * pad - (16 if llm else 0), title_size):
            self.parts.append(svg_text(x + pad, cursor, [line], title_size, stroke, weight="bold"))
            cursor += title_size * 1.22
        cursor += 3
        for text in body:
            for line in wrap(text, w - 2 * pad, size):
                self.parts.append(svg_text(x + pad, cursor, [line], size))
                cursor += size * 1.3
            cursor += 2
        if llm:
            self.parts.append(badge(x + w - 48, y + 6, "LLM"))

    def _draw_cylinder(
        self, x: float, y: float, w: float, h: float, stroke: str, fill: str, title: str, body: list[str], size: float
    ) -> None:
        rx, ry = w / 2, 14
        self.parts.append(
            f'<path d="M{x:g},{y:g} a{rx:g},{ry} 0 0 0 {w:g},0 v{h:g} a{rx:g},{ry} 0 0 1 -{w:g},0 z" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
            f'<ellipse cx="{x + rx:g}" cy="{y:g}" rx="{rx:g}" ry="{ry}" fill="#E5DBFF" stroke="{stroke}" stroke-width="2"/>'
        )
        self.parts.append(svg_text(x + rx, y + 5, [title], size, stroke, "middle", "bold"))
        lines = [line for text in body for line in wrap(text, w - 30, 11.5)]
        self.parts.append(svg_text(x + rx, y + 32, lines, 11.5, INK, "middle", line_height=16))

    def _draw_chips(self, node: Node, x: float, y: float, w: float, stroke: str, title: str, body: list[str]) -> None:
        self.parts.append(svg_text(x + 12, y + 20, [title], 12.5, stroke, weight="bold"))
        names = self.hints.chips[node.id]
        cx, cy = x + 12, y + 34
        for chip in [part.strip() for part in body[-1].split("·")] if body else []:
            color = COLORS.get(names.get(chip, ""), stroke)
            width = 14 + len(chip) * 7
            if cx + width > x + w - 8:
                cx, cy = x + 12, cy + 32
            self.parts.append(
                f'<rect x="{cx:g}" y="{cy:g}" width="{width:g}" height="24" rx="5" fill="#fff" stroke="{color}" stroke-width="1.5"/>'
            )
            self.parts.append(svg_text(cx + width / 2, cy + 16, [chip], 11, color, "middle", "bold"))
            cx += width + 8

    def route(self, edge: Edge, options: dict[str, str]) -> list[tuple[float, float]]:
        """The polyline of an edge, from the source box to the target box."""
        ax, ay, aw, ah = self._box(edge.a)
        bx, by, bw, bh = self._box(edge.b)
        gap = 2  # the arrow head stops on the border
        if "vhv" in options:
            x1, y, x2 = float(options["x1"]), float(options["y"]), float(options["x2"])
            start = ay + ah if y > ay + ah else ay
            end = by - gap if y < by else by + bh + gap
            return [(x1, start), (x1, y), (x2, y), (x2, end)]
        if "hv" in options:
            y1, x2 = float(options["y1"]), float(options["x2"])
            start = ax + aw if x2 > ax + aw else ax
            end = by - gap if y1 < by else by + bh + gap
            return [(start, y1), (x2, y1), (x2, end)]
        if "hvh" in options:
            y1, x, y2 = float(options["y1"]), float(options["x"]), float(options["y2"])
            start = ax + aw if x > ax + aw else ax
            end = bx - gap if x < bx else bx + bw + gap
            return [(start, y1), (x, y1), (x, y2), (end, y2)]
        overlap_x = min(ax + aw, bx + bw) > max(ax, bx)
        overlap_y = min(ay + ah, by + bh) > max(ay, by)
        if "x" in options or (overlap_x and not overlap_y and "y" not in options):
            x = float(options.get("x", (max(ax, bx) + min(ax + aw, bx + bw)) / 2))
            if by >= ay + ah:
                return [(x, ay + ah), (x, by - gap)]
            return [(x, ay), (x, by + bh + gap)]
        if "y" in options or overlap_y:
            y = float(options.get("y", (max(ay, by) + min(ay + ah, by + bh)) / 2))
            if bx >= ax + aw:
                return [(ax + aw, y), (bx - gap, y)]
            return [(ax, y), (bx + bw + gap, y)]
        x1, x2 = ax + aw / 2, bx + bw / 2
        if by >= ay + ah:
            mid = (ay + ah + by) / 2
            return [(x1, ay + ah), (x1, mid), (x2, mid), (x2, by - gap)]
        mid = (by + bh + ay) / 2
        return [(x1, ay), (x1, mid), (x2, mid), (x2, by + bh + gap)]

    def _box(self, ident: str) -> Box:
        if ident not in self.boxes:
            raise MermaidSvgError(f"edge endpoint {ident!r} is not a drawn node or group")
        return self.boxes[ident]

    def draw_edge(self, edge: Edge) -> None:
        options = self.hints.edge.get(edge.key, {})
        color = COLORS.get(options.get("color", ""), "") or self.edge_color(edge.a)
        self.colors.add(color)
        points = self.route(edge, options)
        if edge.both:  # leave room for the reverse arrow head
            (x0, y0), (x1, y1) = points[0], points[1]
            points[0] = (x0, y0 + (2 if y1 > y0 else -2)) if x0 == x1 else (x0 + (2 if x1 > x0 else -2), y0)
        path = "M" + " L".join(f"{x:g},{y:g}" for x, y in points)
        marker = f"url(#arrow-{color.strip('#')})"
        attrs = f'fill="none" stroke="{color}" stroke-width="{1.8 if edge.dashed else 2.2}"'
        if edge.dashed:
            attrs += ' stroke-dasharray="7 5"'
        if edge.arrow or edge.both:
            attrs += f' marker-end="{marker}"'
        if edge.both:
            attrs += f' marker-start="{marker}"'
        self.parts.append(f'<path d="{path}" {attrs}/>')
        if edge.label and options.get("label") != "none":
            self._draw_edge_label(edge.label, options, points, color)

    def _draw_edge_label(
        self, label: str, options: dict[str, str], points: list[tuple[float, float]], color: str
    ) -> None:
        text = " ".join(label_lines(label))
        lines = textwrap.wrap(text, int(options["wrap"])) if "wrap" in options else [text]
        if "label" in options:
            parts = options["label"].split(",")
            x, y = float(parts[0]), float(parts[1])
            anchor = parts[2] if len(parts) > 2 else "start"
        else:  # the middle of the longest segment
            (x0, y0), (x1, y1) = max(
                zip(points, points[1:]), key=lambda seg: abs(seg[0][0] - seg[1][0]) + abs(seg[0][1] - seg[1][1])
            )
            if x0 == x1:
                x, y, anchor = x0 + 8, (y0 + y1) / 2, "start"
            else:
                x, y, anchor = (x0 + x1) / 2, y0 - 8, "middle"
        size = float(options.get("fs", 11.5))
        self.parts.append(svg_text(x, y, lines, size, color, anchor, "bold", halo=True, line_height=13))

    def render(self) -> str:
        assert self.hints.canvas is not None
        width, height, ox, oy = self.hints.canvas
        for group in self.graph.groups.values():
            self.draw_group(group)
        for node in self.graph.nodes.values():
            self.draw_node(node)
        for edge in self.graph.edges:
            self.draw_edge(edge)
        for x, y, text in self.hints.text:
            self.parts.append(svg_text(x, y, [text], 11, GRAY, weight="bold", halo=True))
        markers = "".join(
            f'<marker id="arrow-{color.strip("#")}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
            f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{color}"/></marker>'
            for color in sorted(self.colors)
        )
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{ox:g} {oy:g} {width:g} {height:g}" '
            f'width="{width:g}" height="{height:g}" font-family="{FONT}">'
            f"<defs>{markers}</defs>"
            f'<rect x="{ox:g}" y="{oy:g}" width="{width:g}" height="{height:g}" fill="#fff"/>'
            + "".join(self.parts)
            + "</svg>\n"
        )


def render_block(block: str) -> tuple[str | None, str]:
    """Render one block; return its ``@out`` path and the SVG text.

    Every failure is a :class:`MermaidSvgError`: a malformed hint
    (``at=1``, a bare ``@``, an unclosed quote, ``fs=0``, a route missing a
    coordinate) raises a raw ``ValueError`` / ``IndexError`` / ``KeyError``
    / ``ZeroDivisionError`` deep in the parser or the drawing code, which is
    reported here as a refusal instead of a traceback.
    """
    try:
        graph = parse(block)
        hints = parse_hints(graph.hints)
        return hints.out, Renderer(graph, hints).render()
    except MermaidSvgError:
        raise
    except Exception as exc:  # noqa: BLE001 — any parse/draw failure is a refusal
        raise MermaidSvgError(f"bad Mermaid block or hint: {exc}") from exc


def mermaid_blocks(markdown: str) -> list[str]:
    """Every fenced ``mermaid`` block of a Markdown text."""
    return re.findall(r"^```mermaid\n(.*?)^```", markdown, re.S | re.M)


def outputs_from_text(markdown: str, base_dir: Path) -> list[tuple[Path, str]]:
    """The ``(svg path, svg text)`` of every block of ``markdown`` with an ``@out`` hint.

    ``@out`` paths resolve against ``base_dir`` (the Markdown file's
    directory). Two blocks declaring the same output are refused —
    compared case-insensitively, as a case-insensitive file system (APFS)
    would write both to one file.
    """
    found: list[tuple[Path, str]] = []
    seen: set[str] = set()
    for block in mermaid_blocks(markdown):
        if not re.search(r"^\s*%%\s*@out\s", block, re.M):
            continue
        out, svg = render_block(block)
        svg_path = (base_dir / str(out)).resolve()
        if str(svg_path).casefold() in seen:
            raise MermaidSvgError(f"duplicate @out: {out}")
        seen.add(str(svg_path).casefold())
        found.append((svg_path, svg))
    return found


def outputs(md_path: Path) -> list[tuple[Path, str]]:
    """The ``(svg path, svg text)`` of every block of ``md_path`` with an ``@out`` hint."""
    return outputs_from_text(md_path.read_text(encoding="utf-8"), md_path.parent)


def run(paths: list[Path], check: bool) -> tuple[int, dict[str, object]]:
    """Render (or check) the SVGs of ``paths``; return the exit code and the JSON report."""
    written, stale, fresh = [], [], []
    try:
        for md_path in paths:
            home = md_path.parent.resolve()
            for svg_path, svg in outputs(md_path):
                if svg_path.suffix != ".svg" or not svg_path.is_relative_to(home):
                    # A document-controlled path is never followed outside its
                    # directory: ``render`` is an allow-listed agent command.
                    raise MermaidSvgError(f"@out must be an .svg under {home}: {svg_path}")
                current = svg_path.read_text(encoding="utf-8") if svg_path.exists() else None
                if current == svg:
                    fresh.append(str(svg_path))
                elif check:
                    stale.append(str(svg_path))
                else:
                    svg_path.parent.mkdir(parents=True, exist_ok=True)
                    svg_path.write_text(svg, encoding="utf-8")
                    written.append(str(svg_path))
    except (MermaidSvgError, OSError, UnicodeDecodeError) as exc:
        return 1, {"status": "error", "message": str(exc)}
    if stale:
        return 1, {"status": "stale", "stale": stale, "up_to_date": fresh}
    return 0, {"status": "ok", "written": written, "up_to_date": fresh}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    render = sub.add_parser("render", help="write (or --check) the SVGs of Markdown files")
    render.add_argument("paths", nargs="+", type=Path)
    render.add_argument("--check", action="store_true", help="exit 1 if an SVG is missing or stale")
    args = parser.parse_args(argv)
    code, report = run(args.paths, args.check)
    print(json.dumps(report, indent=2))
    return code


if __name__ == "__main__":
    sys.exit(main())
