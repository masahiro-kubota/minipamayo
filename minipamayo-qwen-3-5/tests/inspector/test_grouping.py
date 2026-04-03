from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from minipamayo_qwen35.inspector.grouping import derive_groups
from minipamayo_qwen35.inspector.models import NormalizedSample


class GroupingTests(unittest.TestCase):
    def test_derive_groups_splits_contiguous_sample_ranges(self) -> None:
        samples = [
            NormalizedSample(
                stage="stage1a_eval",
                run_name="curve_eval",
                sample_id="0001",
                sample_index=10,
                image_path="/tmp/0001.jpeg",
                source_frame_id="20",
                fde_m=1.0,
                command="right",
            ),
            NormalizedSample(
                stage="stage1a_eval",
                run_name="curve_eval",
                sample_id="0002",
                sample_index=11,
                image_path="/tmp/0002.jpeg",
                source_frame_id="22",
                fde_m=2.0,
                command="right",
            ),
            NormalizedSample(
                stage="stage1a_eval",
                run_name="curve_eval",
                sample_id="0003",
                sample_index=20,
                image_path="/tmp/0003.jpeg",
                source_frame_id="40",
                fde_m=3.0,
                command="left",
            ),
            NormalizedSample(
                stage="stage1a_eval",
                run_name="curve_eval",
                sample_id="0004",
                sample_index=21,
                image_path="/tmp/0004.jpeg",
                source_frame_id="42",
                fde_m=4.0,
                command="left",
            ),
        ]

        grouped_samples, groups = derive_groups(samples)

        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0].group_id, "derived:10-11")
        self.assertEqual(groups[1].group_id, "derived:20-21")
        self.assertEqual(groups[0].length, 2)
        self.assertEqual(groups[1].length, 2)
        self.assertEqual(grouped_samples[0].group_frame_index, 0)
        self.assertEqual(grouped_samples[1].group_frame_index, 1)
        self.assertEqual(grouped_samples[2].group_frame_index, 0)
        self.assertEqual(grouped_samples[3].group_frame_index, 1)
        self.assertEqual(grouped_samples[0].group_length, 2)
        self.assertEqual(grouped_samples[2].group_length, 2)


if __name__ == "__main__":
    unittest.main()
