# Stage 2 Notes

`stage2` は `reasoning_sft` の段階です。

整理方針は [REORGANIZATION.md](/home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/stage2/REORGANIZATION.md) を参照してください。

役割:

- `stage1A` で入れた観測 + trajectory token 契約の上に reasoning supervision を足す
- free-running で reasoning を生成し、`<|traj_future_start|>` を境界として `stage1B/expert_cfm` に handoff する

重要な前提:

- `stage2` の挙動を評価するには、先に `stage1A` と `stage1B` がある程度学習できている必要があります
- 特に `stage2` inference の handoff は、`stage1A` が
  - history token
  - future trajectory token
  - special token (`<|cot_start|>`, `<|cot_end|>`, `<|traj_future_start|>`)
  の契約を十分学習していることを前提にします
- `stage1B` が弱い場合も、`stage2 -> expert_cfm` の end-to-end 評価は不安定になります

そのため、`stage2` の smoke で `<|traj_future_start|>` が出ない場合は、すぐに

- `stage2` の設計が誤っている

とは言えません。次の可能性があります。

- `stage1A` の token 契約学習がまだ浅い
- `stage1B` を含む前段の学習量が足りない
- `stage2` 自体の学習量が足りない

実務上の読み方:

- `1 epoch` smoke:
  - 配線確認用
  - train loop / checkpoint / handoff code path が壊れていないかを見る
- `free-running handoff` の成否:
  - `stage2` 単体の品質というより、`stage1A/1B + stage2` の複合結果

現状の優先順位:

1. `stage1A` / `stage1B` の契約を Alpamayo に寄せる
2. その上で `stage2` が free-running で `<|traj_future_start|>` を安定して出すか確認する
3. 失敗時は `stage2` だけでなく前段の学習量不足も疑う
