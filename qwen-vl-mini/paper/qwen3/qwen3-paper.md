# Qwen3 Technical Report

Qwen3に関する論文の要約です。本モデルは、推論（Thinking）モードと非推論（Non-thinking）モードを統合した大規模言語モデル（LLM）ファミリーであり、119言語をサポートし、36兆トークンで事前学習されています。

## アブストラクト (Abstract)
**内容まとめ**:
本論文ではQwen3を紹介します。Qwenモデルファミリーの最新バージョンであり、性能、効率性、多言語能力を向上させるよう設計されています。主な特徴は以下の通りです：
1.  **Thinking/Non-thinking モードの統合**: 複雑な多段階推論用のthinkingモードと、高速な応答用のnon-thinkingモードを単一のモデルに統合。これにより、GPT-4oのようなチャット最適化モデルとQwQ-32Bのような推論特化モデルを切り替える必要がなくなります。
2.  **Thinking Budget機構**: ユーザーが推論時の計算リソースを適応的に割り当て可能。タスクの複雑さに応じてレイテンシと性能のバランスを調整。
3.  **多言語サポート拡大**: Qwen2.5の29言語から119言語・方言に拡大し、グローバルなアクセシビリティを向上。
4.  **Strong-to-Weak Distillation**: フラッグシップモデルの知識を活用し、軽量モデルの構築に必要な計算リソースを大幅削減。
モデルはDense（0.6B/1.7B/4B/8B/14B/32B）とMoE（30B-A3B/235B-A22B）で提供されます。

## 1. イントロダクション (Introduction)
**内容まとめ**:
AGI/ASI実現に向け、GPT-4o、Claude 3.7、DeepSeek-V3などの大規模基盤モデルが大きな進歩を遂げています。特に推論モデル（o3、DeepSeek-R1など）は強化学習を通じた推論時スケーリングの可能性を示しています。
Qwen3は以下の革新を導入します：
- **Thinkingモードとnon-thinkingモードの統合**: ユーザーが単一モデル内で動的にモード切替可能。
- **Thinking Budget**: 推論深度を細かく制御可能。
- **36兆トークン/119言語での事前学習**: 多言語能力の大幅強化。
フラッグシップモデルQwen3-235B-A22Bは、AIME'24で85.7、LiveCodeBench v5で70.7、CodeForcesで2,056点を達成しています。

## 2. アーキテクチャ (Architecture)
**内容まとめ**:
Qwen3シリーズは6つのDenseモデルと2つのMoEモデルで構成されます：
-  **Denseモデル**: Qwen3-0.6B/1.7B/4B/8B/14B/32B
-  **MoEモデル**: Qwen3-30B-A3B（30Bパラメータ、3Bアクティブ）、Qwen3-235B-A22B（235B総パラメータ、22Bアクティブ）

アーキテクチャはQwen2.5と同様で以下を採用：
- **Grouped Query Attention (GQA)**
- **SwiGLU活性化関数**
- **Rotary Positional Embeddings (RoPE)**
- **RMSNorm with pre-normalization**
- **QK-Norm**（Qwen2のQKV-biasを廃止し、安定した学習のため導入）

MoEモデルは128個のエキスパートのうち8個をアクティブ化し、**global-batch load balancing loss**を採用してエキスパートの特化を促進。Qwen2.5-MoEと異なり、共有エキスパートは使用しません。

## 3. 事前学習 (Pre-training)
**内容まとめ**:

*   **3.1 事前学習データ**: 36兆トークン、119言語・方言をカバー。コーディング、STEM、推論、書籍、多言語テキスト、合成データを含む。Qwen2.5-VLでPDFからテキスト抽出し、Qwen2.5-Math/Coderで合成データ生成。
*   **3.2 事前学習ステージ**:
    *   **Stage 1 (General Stage)**: 30兆トークン以上、シーケンス長4,096で訓練。言語能力と一般的な世界知識を構築。
    *   **Stage 2 (Reasoning Stage)**: STEM、コーディング、推論、合成データの比率を増加。約5兆トークンで学習。
    *   **Stage 3 (Long Context Stage)**: シーケンス長32,768に拡張。RoPEのベース周波数を10,000→1,000,000に増加（ABF技術）。YARNとDual Chunk Attention (DCA)で推論時4倍のシーケンス長に対応。
*   **3.3 事前学習評価**: MoEモデルは同等のDenseモデルの1/5のアクティブパラメータで同等性能を達成。Qwen3-235B-A22Bは、DeepSeek-V3を総パラメータ1/3、アクティブパラメータ2/3で上回る。

## 4. ポストトレーニング (Post-training)
**内容まとめ**:
ポストトレーニングは4段階で構成され、2つの目標（Thinking Control、Strong-to-Weak Distillation）を達成します：

*   **4.1 Long-CoT Cold Start**: 数学、コード、論理推論、STEMをカバーするデータセットで、Chain-of-Thoughtパターンの基礎を構築。QwQ-32Bで候補応答を生成し、厳格なフィルタリングを実施。
*   **4.2 Reasoning RL**: 約4,000のクエリ-検証ペアでGRPOアルゴリズムを使用。大規模バッチ、多数のロールアウト、off-policy学習で効率向上。AIME'24スコアが70.1→85.1に向上。
*   **4.3 Thinking Mode Fusion**: 非推論能力を推論モデルに統合。`/think`と`/no_think`フラグでモード切替可能。空のthinkブロックを残す設計で内部フォーマット一貫性を維持。**Thinking Budget**機能が自然創発。
*   **4.4 General RL**: 20以上のタスクをカバーする報酬システムで汎用能力を強化。指示追従、フォーマット追従、嗜好アライメント、エージェント能力を最適化。Rule-based/Model-based報酬を使用。
*   **4.5 Strong-to-Weak Distillation**: 軽量モデル（0.6B〜14Bと30B-A3B）に対して実施。Off-policy蒸留とOn-policy蒸留の2フェーズ。4段階学習比で1/10のGPU時間で同等性能を達成。

## 5. 評価 (Evaluation)
**内容まとめ**:

以下のカテゴリで包括的に評価：
- **General Tasks**: MMLU-Redux、GPQA-Diamond、C-Eval、LiveBench
- **Alignment Tasks**: IFEval、Arena-Hard、AlignBench、Creative Writing、WritingBench
- **Math & Text Reasoning**: MATH-500、AIME'24/25、ZebraLogic、AutoLogi
- **Agent & Coding**: BFCL v3、LiveCodeBench v5、CodeForces
- **Multilingual Tasks**: Multi-IF、INCLUDE、MMMLU、MT-AIME2024、PolyMath、MLogiQA

主要結果：
1.  **Qwen3-235B-A22B (Thinking)**: DeepSeek-R1をアクティブパラメータ60%、総パラメータ35%で17/23ベンチマークで上回る。OpenAI-o1、Gemini2.5-Proと競争的。
2.  **Qwen3-235B-A22B (Non-thinking)**: GPT-4o-2024-11-20を18/23ベンチマークで上回る。
3.  **Qwen3-32B (Thinking)**: QwQ-32Bを17/23ベンチマークで上回り、32Bサイズで最強の推論モデル。OpenAI-o3-mini (medium)と競争的。
4.  **Qwen3-32B (Non-thinking)**: Qwen2.5-72B-Instructと同等以上（パラメータ数1/2以下）。
5.  **軽量モデル**: Strong-to-Weak Distillationの成功により、同等以上のパラメータ数のオープンソースモデルを一貫して上回る。

## 6. 結論 (Conclusion)
**内容まとめ**:
Qwen3はThinkingモードとnon-thinkingモードを備え、ユーザーが複雑な推論タスクに使用するトークン数を動的に管理できます。36兆トークンで事前学習され、119言語をサポートします。コード生成、数学、推論、エージェントタスクで標準ベンチマーク全体にわたり強い性能を示しています。

今後の研究課題：
- より高品質で多様なデータでの事前学習スケールアップ
- 効果的な圧縮や超長コンテキストへのスケーリングのためのアーキテクチャ・学習手法改善
- 環境フィードバックから学習するエージェントベースRLシステムへの計算リソース増加
- 推論時スケーリングを必要とする複雑なタスクに対応可能なエージェント構築
