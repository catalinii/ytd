#!/usr/bin/env python3
"""Focused checks for speaker-anchor tracking (no test framework needed).

Run:  venv/bin/python test_tracking.py
"""
import inspect

from burn_subtitles import (
    build_parser,
    render_captioned_square,
    select_anchored_cx,
)


def check(name, actual, expected):
    assert actual == expected, f"{name}: got {actual!r}, want {expected!r}"
    print(f"ok - {name}")


# nearest face to the anchor wins (podium speaker + keyboard person)
check("two faces pick left speaker",
      select_anchored_cx([260.0, 1160.0], 260), 260.0)
check("two faces pick right person when anchored right",
      select_anchored_cx([260.0, 1160.0], 1160), 1160.0)
# only the wrong person visible -> reject (gap filled from neighbors later)
check("wrong-person-only frame rejected",
      select_anchored_cx([1150.0], 260), None)
check("empty frame rejected", select_anchored_cx([], 260), None)
# tolerance boundary is inclusive
check("tolerance edge kept",
      select_anchored_cx([560.0], 260, anchor_tol=300), 560.0)
check("past tolerance rejected",
      select_anchored_cx([561.0], 260, anchor_tol=300), None)

# 260 is the default everywhere so plain runs lock onto the left speaker
params = inspect.signature(render_captioned_square).parameters
check("render default anchor_x", params["anchor_x"].default, 260)
check("render default anchor_tol", params["anchor_tol"].default, 300)
cli = build_parser().parse_args([])
check("CLI default anchor_x", cli.anchor_x, 260)
check("CLI --no-anchor resolves to None",
      None if build_parser().parse_args(["--no-anchor"]).no_anchor else 260,
      None)

print("all tracking checks passed")
