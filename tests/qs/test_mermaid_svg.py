"""Tests for ``scripts/qs/mermaid_svg.py`` — Mermaid blocks rendered to laid-out SVG."""

from __future__ import annotations

import ast
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


def _module_names(node: ast.Import | ast.ImportFrom) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    assert node.level == 0, "a relative import is not stdlib"
    return [node.module or ""]


def _is_main_guard(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        and len(node.test.comparators) == 1
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value == "__main__"
    )


def test_import_contract_lets_epic_doc_load_mains_renderer() -> None:
    """QS-404 AC 12: ``epic_doc.py land`` execs ``origin/main``'s renderer source.

    That is only safe while the module imports only stdlib modules, never
    reads ``__file__`` / ``__spec__`` / a data file, and does nothing at
    import beyond defining names. Checked on the AST, so a docstring or a
    comment that mentions one of these names never trips the pin.
    """
    source = (REPO_ROOT / "scripts" / "qs" / "mermaid_svg.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for name in _module_names(node):
                assert name.split(".")[0] in sys.stdlib_module_names, name
        if isinstance(node, ast.Name):
            assert node.id not in ("__file__", "__spec__"), node.id
        if isinstance(node, ast.Attribute):
            assert node.attr not in ("__file__", "__spec__"), node.attr
        if isinstance(node, ast.Call):
            func = node.func
            assert not (isinstance(func, ast.Name) and func.id == "open"), "open() call"
            assert not (
                isinstance(func, ast.Attribute)
                and func.attr == "open"
                and isinstance(func.value, ast.Name)
                and func.value.id == "io"
            ), "io.open() call"
    allowed = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)
    for index, stmt in enumerate(tree.body):
        if index == 0 and isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            continue  # the module docstring
        assert isinstance(stmt, allowed) or _is_main_guard(stmt), ast.dump(stmt)[:120]
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and stmt.value is not None:
            # An assignment runs at import: it may only compile a regex.
            for call in (n for n in ast.walk(stmt.value) if isinstance(n, ast.Call)):
                assert _is_re_compile(call), ast.dump(call)[:120]


def _is_re_compile(call: ast.Call) -> bool:
    func = call.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "compile"
        and isinstance(func.value, ast.Name)
        and func.value.id == "re"
    )


def test_import_contract_pin_flags_a_call_at_import() -> None:
    """The pin's own guard: a module-level assignment calling anything but ``re.compile``."""
    (stmt,) = ast.parse("X = open('f')\n").body
    assert isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call)
    assert not _is_re_compile(stmt.value)
    (stmt,) = ast.parse("X = re.compile('x')\n").body
    assert isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call)
    assert _is_re_compile(stmt.value)


def test_outputs_from_text_matches_outputs(tmp_path: Path) -> None:
    """QS-404 AC 13: ``outputs()`` is ``outputs_from_text`` over the file's text and directory."""
    text = f"```mermaid\n{BLOCK}```\n"
    md = tmp_path / "doc.md"
    md.write_text(text, encoding="utf-8")
    from_text = mermaid_svg.outputs_from_text(text, tmp_path)
    assert from_text == mermaid_svg.outputs(md)
    assert from_text == [((tmp_path / "img" / "run.svg").resolve(), mermaid_svg.render_block(BLOCK)[1])]


def test_outputs_from_text_refuses_a_duplicate_out(tmp_path: Path) -> None:
    """QS-404 AC 13: two blocks writing one SVG would silently overwrite each other."""
    text = f"```mermaid\n{BLOCK}```\n\n```mermaid\n{BLOCK}```\n"
    with pytest.raises(mermaid_svg.MermaidSvgError, match="duplicate @out"):
        mermaid_svg.outputs_from_text(text, tmp_path)


def test_outputs_from_text_refuses_a_case_only_duplicate(tmp_path: Path) -> None:
    """Review fix #01 F6: on a case-insensitive file system both blocks would write one file."""
    lower = BLOCK.replace("img/run.svg", "img/view.svg")
    upper = BLOCK.replace("img/run.svg", "img/View.svg")
    text = f"```mermaid\n{lower}```\n\n```mermaid\n{upper}```\n"
    with pytest.raises(mermaid_svg.MermaidSvgError, match="duplicate @out"):
        mermaid_svg.outputs_from_text(text, tmp_path)


_BAD_HINT_BLOCKS = [
    'flowchart TB\n    a["A"]\n    %% @out x.svg\n    %% @canvas 10 10\n    %% @node a at=1 size=1,1\n',
    'flowchart TB\n    a["A<br/>body"]\n    %% @out x.svg\n    %% @canvas 10 10\n    %% @node a at=0,0 size=100,50 fs=0\n',
    'flowchart TB\n    a["A"]\n    %% @\n',
    'flowchart TB\n    a["A"]\n    %% @canvas 100\n',
    'flowchart TB\n    a["A"]\n    %% @canvas 10 10 "unclosed\n',
    (
        'flowchart TB\n    a["A"]\n    b["B"]\n    a --> b\n    %% @canvas 400 400\n'
        "    %% @node a at=0,0 size=10,10\n    %% @node b at=0,100 size=10,10\n    %% @edge a->b vhv y=5 x2=5\n"
    ),
]


@pytest.mark.parametrize("block", _BAD_HINT_BLOCKS)
def test_render_block_wraps_raw_exceptions(block: str) -> None:
    """QS-404 AC 14: a malformed hint is a ``MermaidSvgError``, never a raw traceback."""
    with pytest.raises(mermaid_svg.MermaidSvgError, match="bad Mermaid block or hint"):
        mermaid_svg.render_block(block)


def test_render_block_keeps_its_own_errors_unchanged() -> None:
    with pytest.raises(mermaid_svg.MermaidSvgError) as info:
        mermaid_svg.render_block('flowchart TB\n    a["A"]\n    %% @node a at=0,0 size=1,1\n')
    assert str(info.value) == "missing @canvas hint"


@pytest.mark.parametrize("block", _BAD_HINT_BLOCKS[:2])
def test_run_reports_a_malformed_hint_as_an_error(tmp_path: Path, block: str) -> None:
    md = tmp_path / "doc.md"
    md.write_text(f"```mermaid\n{block}```\n", encoding="utf-8")
    code, report = mermaid_svg.run([md], check=True)
    assert code == 1
    assert report["status"] == "error"
    assert "bad Mermaid block or hint" in str(report["message"])


def test_run_reports_a_non_utf8_svg_as_an_error(tmp_path: Path) -> None:
    md = tmp_path / "doc.md"
    md.write_text(f"```mermaid\n{BLOCK}```\n", encoding="utf-8")
    (tmp_path / "img").mkdir()
    (tmp_path / "img" / "run.svg").write_bytes(b"\xff\xfe\x00garbage")
    code, report = mermaid_svg.run([md], check=True)
    assert code == 1
    assert report["status"] == "error"


@pytest.mark.parametrize("check", [False, True], ids=["render", "check"])
@pytest.mark.parametrize(
    "out",
    ["../escape.svg", "{tmp}/abs.svg", "img/notes.txt", "img/run.SVG.bak"],
    ids=["parent", "absolute", "not-svg", "svg-suffix-inside"],
)
def test_run_refuses_an_out_outside_the_markdown_directory(tmp_path: Path, out: str, check: bool) -> None:
    """Review fix #02 G2: ``render`` writes only ``.svg`` files under the Markdown file's directory."""
    docs = tmp_path / "docs"
    docs.mkdir()
    md = docs / "doc.md"
    out = out.format(tmp=tmp_path)  # an absolute path, still inside the sandbox
    md.write_text(f"```mermaid\n{BLOCK.replace('img/run.svg', out)}```\n", encoding="utf-8")
    code, report = mermaid_svg.run([md], check=check)
    assert code == 1
    assert report["status"] == "error"
    assert "@out must be an .svg under" in str(report["message"])
    assert sorted(p.name for p in tmp_path.rglob("*") if p.is_file()) == ["doc.md"]


def test_run_refuses_an_out_through_a_symlink_leaving_the_directory(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "img").symlink_to(outside, target_is_directory=True)
    md = docs / "doc.md"
    md.write_text(f"```mermaid\n{BLOCK}```\n", encoding="utf-8")
    code, report = mermaid_svg.run([md], check=False)
    assert code == 1 and report["status"] == "error"
    assert list(outside.iterdir()) == []


def test_every_svg_in_docs_is_up_to_date() -> None:
    """A Mermaid block with an ``@out`` hint must have its SVG regenerated after an edit."""
    docs = sorted((REPO_ROOT / "docs").rglob("*.md"))
    code, report = mermaid_svg.run(docs, check=True)
    assert code == 0, f"run: python scripts/qs/mermaid_svg.py render <doc> — {report}"
