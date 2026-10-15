"""Focused layout tests for the header-banner / IG canvas.

Run with: venv/bin/python -m unittest tests.test_header_layout -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from burn_subtitles import (  # noqa: E402
    compute_canvas,
    parse_hex_color,
    render_text_bar,
)


class CanvasGeometryTest(unittest.TestCase):
    def test_square_legacy_unchanged(self):
        self.assertEqual(compute_canvas(720, "square"), (720, 720, 0, 0))
        self.assertEqual(compute_canvas(720, "1:1"), (720, 720, 0, 0))

    def test_feed_4x5_header_on_top(self):
        out_w, out_h, hh, fh = compute_canvas(720, "4:5")
        self.assertEqual((out_w, out_h), (720, 900))
        self.assertEqual((hh, fh), (180, 0))

    def test_reels_9x16_splits_header_footer(self):
        out_w, out_h, hh, fh = compute_canvas(720, "9:16")
        self.assertEqual((out_w, out_h), (720, 1280))
        self.assertEqual(hh + fh, 560)
        self.assertGreater(hh, fh)  # roomier hook on top
        self.assertGreater(fh, 0)

    def test_square_video_pasted_below_header(self):
        # The captioned square must sit exactly under the banner so burn-in
        # captions (anchored to the square's lower third) never overlap it.
        crop_s = 720
        out_w, out_h, hh, fh = compute_canvas(crop_s, "4:5")
        video_top, video_bottom = hh, hh + crop_s
        self.assertEqual((video_top, video_bottom), (180, 900))
        self.assertEqual(video_bottom + fh, out_h)
        self.assertEqual(out_w, crop_s)

    def test_explicit_override_must_sum_to_preset_extra(self):
        with self.assertRaises(ValueError):
            compute_canvas(720, "4:5", header_h=100, footer_h=100)  # != 180
        self.assertEqual(compute_canvas(720, "4:5", header_h=100, footer_h=80),
                         (720, 900, 100, 80))

    def test_unknown_aspect_rejected(self):
        with self.assertRaises(ValueError):
            compute_canvas(720, "16:9")


class BannerBarTest(unittest.TestCase):
    def test_bar_size_matches_canvas_strip(self):
        bar = render_text_bar(720, 180, "Free counseling for life problems")
        self.assertEqual(bar.size, (720, 180))

    def test_banner_text_actually_drawn(self):
        bg = (17, 24, 39)
        bar = render_text_bar(720, 180, "Free counseling for life problems",
                              bg=bg)
        pixels = list(bar.getdata())
        self.assertGreater(len(set(pixels)), 1)  # not a solid fill
        self.assertIn(bg, set(pixels))  # background preserved around text

    def test_empty_text_is_clean_fill(self):
        bg = (17, 24, 39)
        bar = render_text_bar(720, 180, "", bg=bg)
        self.assertEqual(set(bar.getdata()), {bg})

    def test_long_message_wraps_and_stays_inside(self):
        text = ("Free counseling for life problems at our church "
                "every Sunday open to everyone DM us today")
        bar = render_text_bar(720, 180, text)
        self.assertEqual(bar.size, (720, 180))
        self.assertGreater(len(set(bar.getdata())), 1)


class ColorTest(unittest.TestCase):
    def test_hex_forms(self):
        self.assertEqual(parse_hex_color("#ff0000", (0, 0, 0)), (255, 0, 0))
        self.assertEqual(parse_hex_color("00ff00", (0, 0, 0)), (0, 255, 0))
        self.assertEqual(parse_hex_color("0,0,255", (0, 0, 0)), (0, 0, 255))
        self.assertEqual(parse_hex_color(None, (1, 2, 3)), (1, 2, 3))
        self.assertEqual(parse_hex_color("bogus", (1, 2, 3)), (1, 2, 3))


if __name__ == "__main__":
    unittest.main()
