from __future__ import annotations

import unittest

import numpy as np

from minipamayo_qwen35.stage1.diagnostics.diagnose_curve_failures import (
    MORPHOLOGY_LATE_DIVERGENCE,
    MORPHOLOGY_LATE_ONSET,
    MORPHOLOGY_OSCILLATORY,
    MORPHOLOGY_STRAIGHT,
    MORPHOLOGY_WRONG_DIRECTION,
    Stage1ASample,
    build_stage1a_vs_stage1b_rows,
    build_pid_rows,
    classify_failure_morphology,
)


def _make_stage1a_sample(
    *,
    sample_id: str,
    gt_lateral: list[float],
    pred_lateral: list[float],
    fde_m: float = 10.0,
    ade_m: float = 5.0,
    action_mae_kappa: float = 0.02,
) -> Stage1ASample:
    forward = np.arange(len(gt_lateral), dtype=np.float64)
    gt_waypoints = np.stack([forward, np.asarray(gt_lateral, dtype=np.float64)], axis=1)
    pred_waypoints = np.stack([forward, np.asarray(pred_lateral, dtype=np.float64)], axis=1)
    return Stage1ASample(
        sample_id=sample_id,
        command="lanefollow",
        image_path=f"/tmp/{sample_id}.png",
        ade_m=ade_m,
        fde_m=fde_m,
        action_mae_kappa=action_mae_kappa,
        gt_waypoints=gt_waypoints,
        pred_waypoints=pred_waypoints,
        payload={},
    )


class CurveFailureDiagnosisTest(unittest.TestCase):
    def test_classify_straight_through(self) -> None:
        sample = _make_stage1a_sample(
            sample_id="straight",
            gt_lateral=[0.0, 0.3, 0.7, 1.5, 2.5, 3.8, 5.0, 6.2],
            pred_lateral=[0.0, 0.0, 0.1, 0.0, 0.2, 0.1, 0.1, 0.0],
        )
        self.assertEqual(classify_failure_morphology(sample), MORPHOLOGY_STRAIGHT)

    def test_classify_wrong_turn_direction(self) -> None:
        sample = _make_stage1a_sample(
            sample_id="wrong",
            gt_lateral=[0.0, 0.4, 1.0, 2.0, 3.2, 4.5, 5.6, 6.2],
            pred_lateral=[0.0, -0.3, -0.9, -1.8, -3.0, -4.1, -5.1, -5.8],
        )
        self.assertEqual(classify_failure_morphology(sample), MORPHOLOGY_WRONG_DIRECTION)

    def test_classify_oscillatory(self) -> None:
        sample = _make_stage1a_sample(
            sample_id="osc",
            gt_lateral=[0.0, 0.3, 0.8, 1.6, 2.7, 3.9, 5.0, 6.0],
            pred_lateral=[0.0, 0.4, -0.9, 1.3, -1.8, 2.2, -2.6, 3.0],
        )
        self.assertEqual(classify_failure_morphology(sample), MORPHOLOGY_OSCILLATORY)

    def test_classify_late_turn_onset(self) -> None:
        sample = _make_stage1a_sample(
            sample_id="late",
            gt_lateral=[0.0, 0.4, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            pred_lateral=[0.0, 0.0, 0.0, 0.1, 0.2, 0.2, 0.9, 2.1, 4.0, 6.5],
        )
        self.assertEqual(classify_failure_morphology(sample), MORPHOLOGY_LATE_ONSET)

    def test_classify_late_divergence(self) -> None:
        sample = _make_stage1a_sample(
            sample_id="late_div",
            gt_lateral=[0.0, 0.3, 0.8, 1.6, 2.7, 4.0, 5.3, 6.5],
            pred_lateral=[0.0, 0.2, 0.7, 1.5, 2.6, 3.0, 2.8, 2.2],
            fde_m=8.0,
        )
        self.assertEqual(classify_failure_morphology(sample), MORPHOLOGY_LATE_DIVERGENCE)

    def test_join_rows_and_pid_rows(self) -> None:
        stage1a_samples = {
            "sample_a": _make_stage1a_sample(
                sample_id="sample_a",
                gt_lateral=[0.0, 1.0, 2.0, 3.0],
                pred_lateral=[0.0, 0.8, 1.6, 2.4],
                fde_m=3.0,
                ade_m=1.5,
                action_mae_kappa=0.02,
            )
        }
        from minipamayo_qwen35.stage1.diagnostics.diagnose_curve_failures import Stage1BSample

        forward = np.arange(4, dtype=np.float64)
        gt_waypoints = np.stack([forward, np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64)], axis=1)
        pred_waypoints = np.stack([forward, np.asarray([0.0, 0.9, 1.8, 2.7], dtype=np.float64)], axis=1)
        pid_pred_waypoints = np.stack(
            [forward, np.asarray([0.0, 1.0, 1.9, 2.9], dtype=np.float64)],
            axis=1,
        )
        stage1b_samples = {
            "sample_a": Stage1BSample(
                sample_id="sample_a",
                command="lanefollow",
                image_path="/tmp/sample_a.png",
                ade_m=1.2,
                fde_m=2.5,
                max_lateral_error_m=0.3,
                action_mae_kappa=0.01,
                gt_waypoints=gt_waypoints,
                pred_waypoints=pred_waypoints,
                pid_ade_m=1.1,
                pid_fde_m=2.0,
                pid_max_lateral_error_m=0.2,
                pid_action_mae_kappa=0.008,
                pid_pred_waypoints=pid_pred_waypoints,
                payload={},
            )
        }
        morphology_map = {"sample_a": MORPHOLOGY_LATE_DIVERGENCE}
        join_rows = build_stage1a_vs_stage1b_rows(
            stage1a_samples=stage1a_samples,
            stage1b_samples=stage1b_samples,
            morphology_map=morphology_map,
        )
        self.assertEqual(len(join_rows), 1)
        self.assertEqual(join_rows[0]["sample_id"], "sample_a")
        self.assertLess(float(join_rows[0]["delta_fde_m"]), 0.0)

        pid_rows = build_pid_rows(stage1b_samples, morphology_map)
        self.assertEqual(len(pid_rows), 1)
        self.assertLess(float(pid_rows[0]["delta_fde_m"]), 0.0)

