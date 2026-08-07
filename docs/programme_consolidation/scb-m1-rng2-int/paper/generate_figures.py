#!/usr/bin/env python3
"""Generate deterministic publication SVGs from frozen reporting sources."""
from __future__ import annotations

import csv
import hashlib
import json
from html import escape
from pathlib import Path
from typing import Iterable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUTPUT = HERE / "figures"
RESULTS = ROOT / "results_summary.csv"
CHRONOLOGY = ROOT / "chronology.csv"

NAVY = "#17324d"
BLUE = "#2867a5"
PALE_BLUE = "#e8f1f8"
GREEN = "#2f7d5b"
PALE_GREEN = "#e8f4ef"
AMBER = "#9b6a12"
PALE_AMBER = "#faf1df"
RED = "#a63d40"
PALE_RED = "#f8e8e8"
INK = "#1f2933"
MUTED = "#52616f"
LINE = "#aab7c2"
PANEL = "#f7f9fb"
WHITE = "#ffffff"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def svg_start(width: int, height: int, title: str, description: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f"<title id=\"title\">{escape(title)}</title>",
        f"<desc id=\"desc\">{escape(description)}</desc>",
        "<defs>",
        f'<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">',
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{NAVY}"/>',
        "</marker>",
        "</defs>",
        f'<rect width="{width}" height="{height}" fill="{WHITE}"/>',
    ]


def svg_end() -> str:
    return "</svg>\n"


def text(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 18,
    weight: int = 400,
    anchor: str = "middle",
    fill: str = INK,
) -> str:
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}">{escape(value)}</text>'
    )


def lines(
    x: float,
    y: float,
    values: Iterable[str],
    *,
    size: int = 18,
    gap: int = 23,
    weight: int = 400,
    anchor: str = "middle",
    fill: str = INK,
) -> list[str]:
    return [
        text(x, y + index * gap, value, size=size, weight=weight, anchor=anchor, fill=fill)
        for index, value in enumerate(values)
    ]


def box(
    x: float,
    y: float,
    width: float,
    height: float,
    label: Iterable[str],
    *,
    fill: str = PANEL,
    stroke: str = NAVY,
    radius: int = 12,
    title_weight: int = 500,
) -> list[str]:
    values = list(label)
    start = y + height / 2 - ((len(values) - 1) * 22) / 2 + 6
    return [
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{radius}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>',
        *lines(
            x + width / 2,
            start,
            values,
            size=17,
            gap=22,
            weight=title_weight,
        ),
    ]


def arrow(x1: float, y1: float, x2: float, y2: float, *, dashed: bool = False) -> str:
    dash = ' stroke-dasharray="6 5"' if dashed else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{NAVY}" stroke-width="2.5" marker-end="url(#arrow)"{dash}/>'
    )


def write(name: str, content: list[str]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / name).write_text("\n".join(content) + "\n", encoding="utf-8")


def causal_computation() -> None:
    out = svg_start(
        1400,
        390,
        "Mandatory learned causal computation",
        "Visible inputs produce hard z2, a registered transition produces hard z3, "
        "and the answer head reads only z3.",
    )
    out += [text(700, 42, "Mandatory learned causal computation", size=25, weight=500)]
    positions = [
        (55, 130, 210, 105, ["Visible inputs"], PALE_BLUE),
        (330, 130, 220, 105, ["Predicted", "hard z2"], PALE_GREEN),
        (650, 130, 250, 105, ["Registered transition", "hard z2 + d"], PALE_BLUE),
        (1000, 130, 210, 105, ["Predicted", "hard z3"], PALE_GREEN),
        (1250, 130, 120, 105, ["Answer"], PALE_BLUE),
    ]
    for item in positions:
        out += box(*item[:4], item[4], fill=item[5])
    out += [
        arrow(265, 182, 330, 182),
        arrow(550, 182, 650, 182),
        arrow(900, 182, 1000, 182),
        arrow(1210, 182, 1250, 182),
    ]
    out += box(690, 285, 170, 65, ["Registered d"], fill=WHITE, stroke=LINE)
    out += [
        arrow(775, 285, 775, 238),
        text(700, 375, "No encoder, prompt, soft-logit, or true-state bypass", size=17, fill=MUTED),
    ]
    out.append(svg_end())
    write("figure-1-causal-computation.svg", out)


def receipt_carriage() -> None:
    out = svg_start(
        1500,
        560,
        "Receipt-gated abstract state carriage",
        "Exact events and an abstract z2 receipt are verified before authority "
        "derivation, position-indexed generation, and scoring under 16 authorities.",
    )
    out += [text(750, 40, "Receipt-gated abstract state carriage", size=25, weight=500)]
    out += box(45, 100, 220, 75, ["Exact z2 event"], fill=PALE_BLUE)
    out += box(45, 205, 220, 75, ["Exact z3 event"], fill=PALE_BLUE)
    out += box(45, 310, 220, 75, ["Answer digest"], fill=PALE_BLUE)
    out += box(45, 425, 220, 75, ["Abstract z2 receipt"], fill=PALE_GREEN)
    out += box(350, 175, 260, 130, ["Verify exact", "z2 → z3 → answer", "chain"], fill=PALE_BLUE)
    out += box(350, 405, 260, 95, ["Verify abstract", "state class"], fill=PALE_GREEN)
    out += box(700, 245, 220, 105, ["Generation gate", "both must pass"], fill=PALE_AMBER, stroke=AMBER)
    out += box(1000, 245, 220, 105, ["Derive state-class", "authority"], fill=PALE_GREEN)
    out += box(1300, 145, 160, 105, ["RNG2", "fixed draws"], fill=PALE_BLUE)
    out += box(1300, 345, 160, 105, ["Score all", "16 authorities"], fill=PALE_BLUE)
    out += [
        arrow(265, 137, 350, 205),
        arrow(265, 242, 350, 235),
        arrow(265, 347, 350, 265),
        arrow(265, 462, 350, 452),
        arrow(610, 240, 700, 275),
        arrow(610, 452, 700, 320),
        arrow(920, 297, 1000, 297),
        arrow(1220, 297, 1300, 198),
        arrow(1380, 250, 1380, 345),
        text(1110, 400, "Authority excludes receipt bytes", size=16, fill=MUTED),
        text(1110, 424, "and exact execution identity", size=16, fill=MUTED),
        text(750, 535, "Carrier compatibility expresses the abstract state class—not exact execution identity", size=17, fill=MUTED),
    ]
    out.append(svg_end())
    write("figure-2-receipt-gated-carriage.svg", out)


def matched_contrast() -> None:
    out = svg_start(
        1400,
        620,
        "Matched natural and alias contrast",
        "Natural and alias executions share prompt, answer, payload, configuration, "
        "and position-indexed draws; authenticated z2 and state authority switch.",
    )
    out += [text(700, 42, "Matched natural / alias contrast", size=25, weight=500)]
    out += box(80, 100, 500, 90, ["Natural execution", "authenticated z2 = A"], fill=PALE_BLUE)
    out += box(820, 100, 500, 90, ["Alias intervention", "authenticated z2 = B"], fill=PALE_GREEN)
    out += box(80, 245, 500, 210, ["FIXED", "Prompt", "Final answer", "Semantic payload", "49 indexed draws"], fill=PANEL, stroke=LINE)
    out += box(820, 245, 500, 210, ["FIXED", "Prompt", "Final answer", "Semantic payload", "49 indexed draws"], fill=PANEL, stroke=LINE)
    out += box(180, 510, 300, 75, ["Authority A", "carrier signal A"], fill=PALE_BLUE)
    out += box(920, 510, 300, 75, ["Authority B", "carrier signal B"], fill=PALE_GREEN)
    out += [
        arrow(330, 190, 330, 245),
        arrow(1070, 190, 1070, 245),
        arrow(330, 455, 330, 510),
        arrow(1070, 455, 1070, 510),
        arrow(580, 350, 820, 350, dashed=True),
        text(700, 333, "Only intended switching source", size=16, fill=MUTED),
        text(700, 377, "authenticated abstract state", size=16, fill=MUTED),
    ]
    out.append(svg_end())
    write("figure-3-matched-contrast.svg", out)


def evidence_hierarchy() -> None:
    with CHRONOLOGY.open(newline="", encoding="utf-8") as stream:
        chronology = list(csv.DictReader(stream))
    if len(chronology) != 8:
        raise ValueError("chronology denominator changed")
    with RESULTS.open(newline="", encoding="utf-8") as stream:
        results = list(csv.DictReader(stream))
    classifications = [row["classification"] for row in chronology]
    positive_count = sum(
        classification
        in {
            "FORMAL_POSITIVE",
            "IMPLEMENTATION_POSITIVE",
            "FORMAL_PROSPECTIVE_POSITIVE",
            "FORMAL_PROSPECTIVE_PUBLIC_POSITIVE",
        }
        for classification in classifications
    )
    valid_negative_count = classifications.count("VALID_NEGATIVE")
    protected_nonqualification_count = classifications.count(
        "PROTECTED_NON_QUALIFICATION_WITH_TERMINAL_ONTOLOGY_DEFECT"
    )
    closure_count = classifications.count("PROGRAMME_CLOSURE")
    sealed = next(row for row in results if row["population"] == "INT-C1")
    categories = [
        (
            f"Positive stages · {positive_count}",
            "Causal transport, receipts, sampler/channel, public integration",
            PALE_GREEN,
            GREEN,
        ),
        (
            f"Valid negative · {valid_negative_count}",
            "SCB-T1 three-seed robustness requirement was not met",
            PALE_AMBER,
            AMBER,
        ),
        (
            f"Protected non-qualification · {protected_nonqualification_count}",
            "INT-Q1: 127/128 descriptive result; qualification not earned",
            PALE_RED,
            RED,
        ),
        (
            f"Programme closure · {closure_count}",
            "Current experimental lineage closed without INT-C1 authority",
            PALE_BLUE,
            BLUE,
        ),
        (
            "Sealed / untested",
            f"INT-C1: {sealed['total_closures']} closures; zero execution",
            PANEL,
            MUTED,
        ),
    ]
    out = svg_start(
        1400,
        620,
        "Evidence hierarchy",
        "Five distinct evidence classes separate formal positives, scientific "
        "negatives, invalid stages, descriptive evidence, and sealed populations.",
    )
    out += [text(700, 42, "Evidence hierarchy", size=25, weight=500)]
    y = 90
    for index, (heading, detail, fill, stroke) in enumerate(categories, start=1):
        out += [
            f'<circle cx="110" cy="{y + 42}" r="27" fill="{fill}" stroke="{stroke}" stroke-width="2"/>',
            text(110, y + 49, str(index), size=18, weight=500, fill=INK),
            f'<rect x="170" y="{y}" width="1120" height="84" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="2"/>',
            text(210, y + 33, heading, size=19, weight=500, anchor="start"),
            text(210, y + 62, detail, size=17, anchor="start", fill=MUTED),
        ]
        y += 101
    out += [text(700, 606, "Categories are not pooled or treated as a quantitative meta-analysis", size=16, fill=MUTED)]
    out.append(svg_end())
    write("figure-4-evidence-hierarchy.svg", out)


def result_table() -> None:
    with RESULTS.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if [row["population"] for row in rows] != ["INT-V1", "INT-Q1", "INT-C1"]:
        raise ValueError("result rows changed")
    labels = {
        "INT-V1": ("Public validation", "128 / 128", "Prospectively validated", PALE_GREEN, GREEN),
        "INT-Q1": ("Protected qualification", "127 / 128", "Not qualified", PALE_AMBER, AMBER),
        "INT-C1": ("Protected confirmation", "0 / 256 executed", "Not tested · sealed", PANEL, MUTED),
    }
    out = svg_start(
        1400,
        480,
        "Public and protected outcomes",
        "Public validation passed 128 of 128, protected qualification did not "
        "qualify at 127 of 128, and protected confirmation was not tested.",
    )
    out += [text(700, 42, "Public and protected outcomes", size=25, weight=500)]
    out += [
        f'<rect x="60" y="80" width="1280" height="55" fill="{NAVY}"/>',
        text(95, 114, "Role", size=17, weight=500, anchor="start", fill=WHITE),
        text(720, 114, "Complete result", size=17, weight=500, fill=WHITE),
        text(1280, 114, "Formal status", size=17, weight=500, anchor="end", fill=WHITE),
    ]
    y = 150
    for row in rows:
        role, value, status, fill, stroke = labels[row["population"]]
        out += [
            f'<rect x="60" y="{y}" width="1280" height="82" fill="{fill}" stroke="{WHITE}" stroke-width="2"/>',
            text(95, y + 49, role, size=18, weight=500, anchor="start"),
            text(720, y + 49, value, size=20, weight=500),
            text(1280, y + 49, status, size=18, weight=500, anchor="end", fill=stroke),
        ]
        y += 84
    out += [
        text(70, 432, "Public and protected denominators are separate and must not be pooled.", size=16, anchor="start", fill=MUTED),
        text(70, 457, "INT-Q1 includes one computation-fidelity failure and two dependent attack failures.", size=16, anchor="start", fill=MUTED),
    ]
    out.append(svg_end())
    write("table-3-public-protected-outcomes.svg", out)


def main() -> None:
    causal_computation()
    receipt_carriage()
    matched_contrast()
    evidence_hierarchy()
    result_table()
    outputs = sorted(OUTPUT.glob("*.svg"))
    manifest = {
        "schema": "exp33a.csw_am_scb_m1_rng2_int_publication_figures.v1",
        "presentation_only": True,
        "frozen_evidence_modified": False,
        "scientific_evidence_added": False,
        "sources": {
            "../results_summary.csv": sha(RESULTS),
            "../chronology.csv": sha(CHRONOLOGY),
            "../paper/PAPER_DRAFT.md": sha(HERE / "PAPER_DRAFT.md"),
        },
        "outputs": {path.name: sha(path) for path in outputs},
    }
    (OUTPUT / "FIGURE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"generated": len(outputs), "output": str(OUTPUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
