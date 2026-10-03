"""Tests for ``scripts/qs/mermaid_svg.py`` — Mermaid blocks rendered to laid-out SVG."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "qs"))

import mermaid_svg  # type: ignore[import-not-found]  # noqa: E402

BLOCK = """flowchart TB
    you["✋ You<br/>single voice"]
    go(["start"])
    subgraph RUN["A run"]
        orch["<b>Orchestrator · LLM</b><br/>event loop"]
        rev{"review"}
    end
    db[("state<br/>runs · tasks")]
    q[["queue<br/>report · alert · question · discovered · finding"]]

    you <-->|"plan · decisions"| orch
    orch -->|"pop + ack"| q
    orch --> db
    rev -.-> q
    q --- db
    you -.- RUN
    classDef human fill:#FFF4E6,stroke:#E8590C
    classDef llm fill:#EDF2FF,stroke:#3B5BDB,stroke-dasharray: 5 4
    class you human
    class orch,rev llm
    style RUN fill:#FAFBFF,stroke:#3B5BDB,stroke-dasharray: 5 4
    %% a plain comment
    %% @out img/run.svg
    %% @canvas 800 500 origin=0,-20
    %% @group RUN at=200,20 size=400,200 title=bottom badge=PY
    %% @node you at=20,40 size=120,60
    %% @node go at=20,420 size=100,40
    %% @node orch at=220,40 size=180,90 fs=11 tfs=14
    %% @node rev at=420,40 size=120,80
    %% @node db at=40,300 size=160,80
    %% @node q at=300,300 size=200,90
    %% @chips q report=purple alert=green
    %% @edge you<->orch label=80,120,middle wrap=8
    %% @edge orch->q vhv x1=300 y=260 x2=350
    %% @edge q->db label=none
    %% @text 20,480 "a note"
"""


def test_parse_reads_nodes_groups_edges_and_styles() -> None:
    graph = mermaid_svg.parse(BLOCK)
    assert graph.nodes["orch"].lines == ["Orchestrator · LLM", "event loop"]
    assert graph.nodes["db"].shape == "cyl"
    assert graph.nodes["q"].shape == "queue"
    assert graph.nodes["rev"].shape == "diamond"
    assert graph.nodes["go"].shape == "stadium"
    assert graph.nodes["orch"].cls == "llm"
    assert graph.groups["RUN"].title == "A run"
    assert graph.groups["RUN"].style["stroke"] == "#3B5BDB"
    keys = [edge.key for edge in graph.edges]
    assert keys == ["you<->orch", "orch->q", "orch->db", "rev->q", "q->db", "you->RUN"]
    assert graph.edges[3].dashed and graph.edges[3].arrow
    assert not graph.edges[4].arrow
    assert len(graph.hints) == 14


def test_parse_reads_chained_edges() -> None:
    graph = mermaid_svg.parse('flowchart TB\n    a["A"]\n    a --> b -.->|"x"| c --- d\n')
    assert [(e.key, e.label, e.dashed, e.arrow) for e in graph.edges] == [
        ("a->b", None, False, True),
        ("b->c", "x", True, True),
        ("c->d", None, False, False),
    ]


def test_render_block_draws_everything() -> None:
    out, svg = mermaid_svg.render_block(BLOCK)
    assert out == "img/run.svg"
    assert svg.startswith('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -20 800 500"')
    for expected in (
        "A run",
        "Orchestrator",
        ">LLM<",
        ">PY<",
        "<polygon",
        "<ellipse",
        ">report<",
        "a note",
        "pop + ack",
    ):
        assert expected in svg
    assert "plan ·" in svg  # wrapped label
    assert "M300,130 L300,260 L350,260 L350,298" in svg  # the vhv route
    assert 'stroke-dasharray="7 5"' in svg
    assert "marker-start=" in svg
    assert 'rx="20"' in svg  # the stadium's rounded ends


@pytest.mark.parametrize(
    ("a", "b", "options", "expected"),
    [
        ((0, 0, 100, 50), (0, 100, 100, 50), {}, [(50, 50), (50, 98)]),
        ((0, 100, 100, 50), (0, 0, 100, 50), {}, [(50, 100), (50, 52)]),
        ((0, 0, 100, 50), (200, 0, 100, 50), {}, [(100, 25), (198, 25)]),
        ((200, 0, 100, 50), (0, 0, 100, 50), {}, [(200, 25), (102, 25)]),
        ((0, 0, 100, 50), (200, 100, 100, 50), {}, [(50, 50), (50, 75), (250, 75), (250, 98)]),
        ((200, 100, 100, 50), (0, 0, 100, 50), {}, [(250, 100), (250, 75), (50, 75), (50, 52)]),
        ((0, 0, 100, 50), (0, 100, 100, 50), {"x": "30"}, [(30, 50), (30, 98)]),
        ((0, 0, 100, 50), (200, 0, 100, 50), {"y": "10"}, [(100, 10), (198, 10)]),
        (
            (0, 0, 100, 50),
            (200, 100, 100, 50),
            {"hv": "yes", "y1": "20", "x2": "250"},
            [(100, 20), (250, 20), (250, 98)],
        ),
        ((200, 0, 100, 50), (0, 0, 100, 40), {"hv": "yes", "y1": "60", "x2": "50"}, [(200, 60), (50, 60), (50, 42)]),
        (
            (0, 0, 100, 50),
            (200, 100, 100, 50),
            {"hvh": "yes", "y1": "20", "x": "150", "y2": "120"},
            [(100, 20), (150, 20), (150, 120), (198, 120)],
        ),
        (
            (200, 0, 100, 50),
            (0, 100, 100, 50),
            {"hvh": "yes", "y1": "20", "x": "350", "y2": "120"},
            [(300, 20), (350, 20), (350, 120), (102, 120)],
        ),
        (
            (0, 100, 100, 50),
            (200, 100, 100, 50),
            {"vhv": "yes", "x1": "50", "y": "10", "x2": "250"},
            [(50, 100), (50, 10), (250, 10), (250, 98)],
        ),
        (
            (0, 0, 100, 50),
            (200, 0, 100, 50),
            {"vhv": "yes", "x1": "50", "y": "80", "x2": "250"},
            [(50, 50), (50, 80), (250, 80), (250, 52)],
        ),
    ],
)
def test_route(a, b, options, expected) -> None:
    graph = mermaid_svg.Graph()
    renderer = mermaid_svg.Renderer(graph, mermaid_svg.Hints(canvas=(10, 10, 0, 0)))
    renderer.boxes = {"a": a, "b": b}
    edge = mermaid_svg.Edge("a", "b", both=False, dashed=False, arrow=True, label=None)
    assert renderer.route(edge, options) == expected


def test_label_without_hint_sits_on_the_longest_segment() -> None:
    block = """flowchart TB
    a["A"]
    b["B"]
    c["C"]
    a -->|"down"| b
    a -->|"across"| c
    %% @canvas 400 400
    %% @node a at=0,0 size=100,50
    %% @node b at=0,200 size=100,50
    %% @node c at=300,0 size=100,50
"""
    _, svg = mermaid_svg.render_block(block)
    assert '<text x="58" y="124"' in svg  # vertical: beside the line
    assert '<text x="199" y="17"' in svg  # horizontal: above the line


@pytest.mark.parametrize(
    ("block", "message"),
    [
        ('flowchart TB\n    a["A"] --> b\n', "unsupported Mermaid line"),
        ("flowchart TB\n    a --> b junk\n", "unsupported Mermaid line"),
        ("flowchart TB\n    -->\n", "unsupported Mermaid line"),
        ("flowchart TB\n    end\n", "'end' without a subgraph"),
        ('flowchart TB\n    subgraph G["g"]\n', "never closed"),
        ("flowchart TB\n    class x foo\n", "unknown node"),
        ('flowchart TB\n    a["A"]\n    %% @node a at=0,0 size=1,1\n', "missing @canvas"),
        ('flowchart TB\n    a["A"]\n    %% @canvas 10 10\n', "missing @node hint"),
        ('flowchart TB\n    a["A"]\n    %% @canvas 10 10\n    %% @nope\n', "unknown layout hint"),
        (
            'flowchart TB\n    a["A"]\n    a --> z\n    %% @canvas 10 10\n    %% @node a at=0,0 size=1,1\n',
            "not a drawn node",
        ),
    ],
)
def test_refusals(block: str, message: str) -> None:
    with pytest.raises(mermaid_svg.MermaidSvgError, match=message):
        mermaid_svg.render_block(block)


def test_cli_renders_then_checks(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    md = tmp_path / "doc.md"
    md.write_text(f'# Doc\n\n```mermaid\n{BLOCK}```\n\n```mermaid\nflowchart TB\n    x["no out hint"]\n```\n')
    assert mermaid_svg.main(["render", "--check", str(md)]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "stale"
    assert mermaid_svg.main(["render", str(md)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["written"] == [str((tmp_path / "img" / "run.svg").resolve())]
    assert mermaid_svg.main(["render", "--check", str(md)]) == 0
    assert json.loads(capsys.readouterr().out)["up_to_date"] == report["written"]
    md.write_text('```mermaid\nflowchart TB\n    a["A"]\n    %% @out x.svg\n```\n')
    assert mermaid_svg.main(["render", str(md)]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "error"


def test_every_svg_in_docs_is_up_to_date() -> None:
    """A Mermaid block with an ``@out`` hint must have its SVG regenerated after an edit."""
    docs = sorted((REPO_ROOT / "docs").rglob("*.md"))
    code, report = mermaid_svg.run(docs, check=True)
    assert code == 0, f"run: python scripts/qs/mermaid_svg.py render <doc> — {report}"
