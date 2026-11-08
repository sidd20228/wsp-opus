"""Scorers of the open-answer experiments 1 and 2 (keyword mapping described in each experiment's README)."""
import re

GAMES = [("freeway", ["freeway", "frogger", "road-crossing", "road crossing", "traffic-crossing",
                      "traffic crossing", "crossy", "chicken cross"]),
         ("breakout", ["breakout", "brick", "arkanoid"]),
         ("pacman", ["pac-man", "pacman", "pac man", "ms. pac", "maze chase", "maze-chase"]),
         ("pong", ["pong", "paddle ball", "squash"])]
SHAPES = [("circle", ["circle", "circular", "disc", "disk", "round", "oval", "ring", "ellipse", "annulus"]),
          ("square", ["square", "rectangle", "rectangular", "box"]),
          ("triangle", ["triangle", "triangular", "pyramid", "wedge"])]


def _section(text, head, nxt):
    # accepts "Game: x", "**Game:** x", "## Game" / "## Game:" headings with the answer on the next lines
    t = text.replace("**", "")
    h = r"^[ \t]*(?:#+[ \t]*)?"
    m = re.search(rf"(?im){h}{head}[ \t]*:?[ \t]*(.*?)(?={h}(?:{nxt})[ \t]*(?::|$)|\Z)", t, re.S)
    return m.group(1).strip() if m else ""


def confidence(text):
    s = _section(text, "Confidence", "$^")
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else None


def _first(text, table):
    low = text.lower()
    best = None
    for label, keys in table:
        for k in keys:
            i = low.find(k)
            if i >= 0 and (best is None or i < best[0]):
                best = (i, label)
    return best[1] if best else None


def score_exp1(text):
    line = _section(text, "Game", "Description|Confidence|Reasoning|Observations")
    line = line.splitlines()[0] if line else ""
    return {"answer_line": line, "guess": _first(line, GAMES), "confidence": confidence(text)}


def score_exp2(text):
    sec = _section(text, "Images", "Example|Confidence")
    ex = _section(text, "Example", "Confidence")
    grid = [ln.strip().strip("`") for ln in ex.splitlines() if re.fullmatch(r"[`\s]*[#.]{8,16}[`\s]*", ln)]
    filled = None
    if len(grid) >= 8:
        # filled if some row has white pixels with a black gap between white runs only at the edges of the shape
        inner = sum(1 for r in grid if re.search(r"#\.+#", r))
        filled = inner <= 1
    return {"images": sec[:400], "guess": _first(sec, SHAPES), "confidence": confidence(text),
            "example_rows": len(grid), "example_filled": filled}
