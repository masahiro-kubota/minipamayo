# Docs Layout

`minipamayo-qwen-3-5/docs` は役割別に次の 3 系統へ整理する。

- `design/`: 安定した設計、契約、構造変更計画
- `experiments/`: 実験計画、結果記録、ベンチマークの実施メモ
- `notes/`: 監査メモ、調査ログ、学習メモ

主な入口:

- `design/`
  - `stage1-stage2-ideal-architecture.md`
  - `alpamayo-alignment-plan.md`
  - `stage3-post-training-design.md`
- `experiments/`
  - `ignore-rule-run-entrypoints.md`
  - `stage1b-lateral-pid-experiment-plan.md`
  - `stage1b-lateral-pid-experiment-results.md`
  - `stage1-stage2-ignore-rule-completion-run-plan.md`
  - `stage1-stage2-ignore-rule-completion-run-results.md`
  - `stage2-inference-speedup-experiment-plan.md`
  - `stage2-inference-speedup-experiment-results.md`
- `notes/`
  - `alpamayo-inference-alignment-audit.md`
  - `alpamayo-stage1-training-notes.md`
  - `stage1-curve-eval-interpretation.md`
  - `stage1-stage2-inference-latency-notes.md`
