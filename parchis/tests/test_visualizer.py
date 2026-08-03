#!/usr/bin/env python3
"""
Regression tests for ParchisVisualizer's board geometry.

The board is now the real photo at docs/images/foto_parchis.png, rendered
as a background image; position_coords/home_coords are a straight,
pixel-calibrated grid measured directly off that photo (see
visualizer.py's ARM_CELL_PITCH / HUB_PIXEL / ARM_COLOR_BY_ROTATION_STEP
and docs/CODE_REVIEW.md for the history of the earlier procedurally-drawn,
curved-template board this replaced). These tests check the geometry
still holds together -- distinct cells, everything within the image
bounds, nothing colliding with a base circle -- not the exact pixel
values, so they keep passing across small recalibrations.
"""

import math

import matplotlib
matplotlib.use('Agg')  # headless, no windows during tests

from parchis.visualization.visualizer import ParchisVisualizer
from parchis.game.board import Board


def _all_points(viz):
    """(label, x, y) for every main-track and home-lane cell."""
    points = [(f'track-{p}', x, y) for p, (x, y) in viz.position_coords.items()]
    for color, cells in viz.home_coords.items():
        points += [(f'{color}-{p}', x, y) for p, (x, y) in cells.items()]
    return points


def test_all_coordinates_within_board_bounds():
    """Every cell (main track and all 4 home lanes, including each
    color's finish at the hub) must render inside the photo's bounds."""
    print("\nTesting all coordinates stay within the board image's bounds...")
    viz = ParchisVisualizer()
    margin = 15  # a bit more than a piece radius, so pieces don't clip the edge
    bad = [
        (label, x, y) for label, x, y in _all_points(viz)
        if not (margin <= x <= viz.IMAGE_WIDTH - margin and margin <= y <= viz.IMAGE_HEIGHT - margin)
    ]
    assert not bad, f"Cells outside the board image's bounds: {bad}"
    print(f"✓ All {len(_all_points(viz))} cells stay within the board bounds")


def test_main_track_has_68_unique_positions():
    """The main track must have exactly 68 positions, each at a distinct coordinate."""
    print("\nTesting main track has 68 distinct positions...")
    viz = ParchisVisualizer()
    assert len(viz.position_coords) == 68

    seen = {}
    for pos, coord in viz.position_coords.items():
        for other_pos, other_coord in seen.items():
            assert math.hypot(coord[0] - other_coord[0], coord[1] - other_coord[1]) > 1e-6, (
                f"Positions {pos} and {other_pos} land at the exact same coordinate "
                f"{coord} -- a regression of the quadrant-boundary coincidence bug "
                f"(an earlier version made e.g. position 55 and 56 identical)"
            )
        seen[pos] = coord
    print("✓ All 68 main-track positions are at distinct coordinates")


def test_home_lanes_are_color_specific():
    """
    Regression test for the shared-home-column bug: all 4 colors must
    have their own coordinate for each logical home position (69-76),
    and those coordinates must differ from each other -- except position
    76 (the finish), which is deliberately the same physical point (the
    hub) for every color.
    """
    print("\nTesting home lanes are private per color...")
    viz = ParchisVisualizer()

    for color in viz.COLORS:
        assert len(viz.home_coords[color]) == 8, f"{color} should have 8 home-lane cells"

    for pos in range(Board.HOME_COLUMN_START, Board.FINAL_POSITION):
        coords_this_pos = {color: viz.home_coords[color][pos] for color in viz.COLORS}
        colors = list(coords_this_pos)
        for i in range(len(colors)):
            for j in range(i + 1, len(colors)):
                c1, c2 = coords_this_pos[colors[i]], coords_this_pos[colors[j]]
                d = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
                assert d > 10, (
                    f"Position {pos}: {colors[i]} and {colors[j]} home-lane cells are "
                    f"only {d:.2f}px apart ({c1} vs {c2}) -- too close to render distinctly"
                )
    print("✓ Every color's home lane is at its own distinct coordinates (finish excluded)")


def test_all_colors_finish_at_the_same_hub_point():
    """Position 76 (the finish) is one physical point on the real board --
    every color's home lane should resolve it to the same coordinate."""
    print("\nTesting all 4 colors' finish position is the same point...")
    viz = ParchisVisualizer()
    finishes = {color: viz.home_coords[color][Board.FINAL_POSITION] for color in viz.COLORS}
    values = list(finishes.values())
    for v in values[1:]:
        assert v == values[0], f"Finish positions differ: {finishes}"
    print("✓ All 4 colors finish at the same hub coordinate")


def test_no_track_or_home_cell_overlaps_a_base_circle():
    """No main-track or home-lane cell should render inside any corner's
    base circle (measured radius: ParchisVisualizer.BASE_CIRCLE_RADIUS_PIXEL)."""
    print("\nTesting no track/home cell overlaps a corner base circle...")
    viz = ParchisVisualizer()
    clearance = 10  # roughly a piece radius

    bad = []
    for color, (cx, cy) in viz.base_positions.items():
        for label, x, y in _all_points(viz):
            d = math.hypot(x - cx, y - cy)
            if d < viz.BASE_CIRCLE_RADIUS_PIXEL + clearance:
                bad.append((label, color, round(d, 2)))

    assert not bad, f"Cells overlapping a base circle: {bad}"
    print("✓ No main-track or home-lane cell overlaps any corner's base circle")


def test_start_positions_are_nearest_their_own_corner():
    """Each color's starting square should be geometrically closest to its own base."""
    print("\nTesting each color's start position is nearest its own corner...")
    viz = ParchisVisualizer()

    for color in viz.COLORS:
        start_pos = Board.STARTING_POSITIONS[color]
        sx, sy = viz.position_coords[start_pos]

        own_dist = math.hypot(sx - viz.base_positions[color][0], sy - viz.base_positions[color][1])
        for other_color, (ox, oy) in viz.base_positions.items():
            if other_color == color:
                continue
            other_dist = math.hypot(sx - ox, sy - oy)
            assert own_dist < other_dist, (
                f"{color}'s start (position {start_pos}) is closer to {other_color}'s "
                f"base ({other_dist:.2f}) than its own ({own_dist:.2f})"
            )
    print("✓ Every color's start square is nearest its own corner")


def test_home_entry_points_are_at_their_arms_outer_edge():
    """Each color's home-entry point (Board.HOME_ENTRY_POINTS) is the "turn"
    cell where the main track, having run the length of that color's arm,
    turns onto the home-lane row -- on the real board this is the cell
    furthest from the hub in its arm (see visualizer.py's
    _build_track_template), not a cell near the center."""
    print("\nTesting home-entry points sit at their arm's outer edge...")
    viz = ParchisVisualizer()
    hub = viz._pixel_to_data(*viz.HUB_PIXEL)
    max_arm_reach = viz._arm_cell_offset(8)

    for color in viz.COLORS:
        entry_pos = Board.HOME_ENTRY_POINTS[color]
        ex, ey = viz.position_coords[entry_pos]
        dist_to_hub = math.hypot(ex - hub[0], ey - hub[1])
        assert dist_to_hub > max_arm_reach - 5, (
            f"{color}'s home entry (position {entry_pos}) is only {dist_to_hub:.2f}px "
            f"from the hub -- expected it at the arm's far edge (~{max_arm_reach:.2f}px)"
        )
    print("✓ All 4 home-entry points sit at their arm's outer edge")


if __name__ == '__main__':
    test_all_coordinates_within_board_bounds()
    test_main_track_has_68_unique_positions()
    test_home_lanes_are_color_specific()
    test_all_colors_finish_at_the_same_hub_point()
    test_no_track_or_home_cell_overlaps_a_base_circle()
    test_start_positions_are_nearest_their_own_corner()
    test_home_entry_points_are_at_their_arms_outer_edge()
    print("\nAll visualizer geometry tests passed!")
