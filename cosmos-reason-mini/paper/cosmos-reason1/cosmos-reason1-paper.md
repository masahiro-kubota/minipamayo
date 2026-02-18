# Cosmos-Reason1: From Physical Common Sense To Embodied Reasoning

Cosmos-Reason1に関する論文の要約です。本モデルは、物理世界を理解し、長いChain-of-Thought推論プロセスを通じて適切な具現化された決定（次のアクションなど）を自然言語で生成できるマルチモーダル大規模言語モデルです。

## アブストラクト (Abstract)
**内容まとめ**:
Physical AIシステムは、物理世界を知覚・理解し、複雑なアクションを実行する必要があります。本論文では、Cosmos-Reason1モデルを紹介します。主な特徴：
1.  **Physical AI推論能力の定義**: 物理的常識（Physical Common Sense）と具現化推論（Embodied Reasoning）の2つの重要な能力を定義。
2.  **階層的オントロジー**: 空間（Space）、時間（Time）、基本物理学（Fundamental Physics）の3カテゴリ、16のサブカテゴリで物理的常識を表現。
3.  **2つのモデル**: Cosmos-Reason1-7BとCosmos-Reason1-56Bを開発。
4.  **2段階訓練**: Physical AI SFT（教師あり微調整）とPhysical AI RL（強化学習）で訓練。
5.  **包括的ベンチマーク**: 物理的常識とembodied推論を評価するベンチマークを構築。
コードとモデルはNVIDIA Open Model Licenseで公開。

## 1. イントロダクション (Introduction)
**内容まとめ**:
LLMはコーディングや数学などの領域で顕著な推論能力を発揮していますが、知識を物理世界に接地する能力に課題があります。本論文では：
- **物理的常識**: 環境の一般的・エンボディメント非依存的理解
- **具現化推論**: 物理環境との将来のインタラクションを知覚・推論・計画する能力
の2つの能力をPhysical AIに必要な基本能力として定義します。
Cosmos-Reason1は動画入力を通じて物理世界を知覚し、長いCoT思考プロセスを経てから応答を生成します。応答には説明的な洞察と具現化された決定（次のアクションなど）が含まれます。

## 2. Physical AI推論 (Physical AI Reasoning)
**内容まとめ**:

*   **2.1 物理的常識推論 (Common Sense Reasoning)**: 物理的常識を3つの大カテゴリと16のサブカテゴリで定義：
    *   **Space（空間）**: Relationship（空間関係）、Plausibility（妥当性）、Affordance（アフォーダンス）、Environment（環境）
    *   **Time（時間）**: Actions（アクション）、Order（順序）、Causality（因果関係）、Camera（カメラ）、Planning（計画）
    *   **Fundamental Physics（基本物理学）**: Attributes（属性）、States（状態）、Object Permanence（物体の永続性）、Mechanics（力学）、Electromagnetism（電磁気学）、Thermodynamics（熱力学）、Anti-Physics（反物理学）
*   **2.2 具現化推論 (Embodied Reasoning)**: 4つの重要能力と5つのエンボディメント（人間、動物、ロボットアーム、ヒューマノイド、自律走行車）で2次元オントロジーを定義：
    *   **Complex Sensory Inputs処理**: 生の感覚入力から意味あるパターンを抽出
    *   **Action Effects予測**: 行動の物理的結果を予測
    *   **Physical Constraints尊重**: 慣性、摩擦、材料特性などを考慮した長期行動計画
    *   **Interactionからの学習**: 環境との相互作用に基づく理解の継続的更新（将来の課題）

## 3. モデルアーキテクチャ (Cosmos-Reason1)
**内容まとめ**:

*   **3.1 マルチモーダルアーキテクチャ**: LLaVA/NVLM-Dと同様のdecoder-onlyアーキテクチャを採用。Vision Encoder → Projector（2層MLPでダウンサンプリング）→ LLMバックボーンの構成。
    *   **Cosmos-Reason1-7B**: Qwen2.5-VLをベースに動的画像/動画処理
    *   **Cosmos-Reason1-56B**: InternViT-300M-V2.5 + Nemotron-H（ハイブリッドMamba-MLP-Transformer）。448×448の最大32フレーム、2fps。
*   **3.2 ハイブリッドMamba-MLP-Transformerバックボーン**: Transformerの自己注意の二次計算量問題に対し、Mambaの線形時間シーケンスモデリングを導入。長コンテキストモデリングのためにTransformer層も一部組み込み。

## 4. 強化学習 (Reinforcement Learning)
**内容まとめ**:

*   **4.1 アルゴリズム**: GRPO（Group Relative Policy Optimization）を採用。別の批評家モデルを訓練・維持する必要がなく、シンプルで計算効率が高い。応答グループ内で報酬を正規化してadvantage関数を導出。
*   **4.2 訓練フレームワーク**: 完全非同期かつ高ロバストなRL訓練フレームワークを構築。ポリシー訓練とアクターロールアウトの異種配置戦略により、約160%の訓練効率向上。ノード障害時も自動再構成可能。

## 5. データ (Data)
**内容まとめ**:

*   **5.1 Physical AI SFT**: 約400万アノテーションをキュレーション。
    *   **5.1.1 物理的常識SFT**: 人間キュレーション動画 → 詳細キャプション → QAペア構築 → DeepSeek-R1で推論トレース抽出 → クリーニング&リライト
    *   **5.1.2 具現化推論SFT**: BridgeData V2、RoboVQA、AgiBot、HoloAssist、自律走行車データからキュレーション。短期間セグメント抽出 → 状態-行動コンテキストアノテーション → 推論QAペア作成 → 推論トレース抽出
    *   **5.1.3 直観的物理学SFT**: 空間連続性（Spatial Puzzles：シャッフルパッチ）、時間の矢（Arrow-of-Time：順逆再生）、物体の永続性（Object Permanence：シミュレーション）の3タスク
*   **5.2 Physical AI RL**: SFTデータソースからMCQ（選択問題）に変換し、ルールベース・検証可能な報酬を実現。30,304問のRLデータセット。

## 6. ベンチマーク (Benchmark)
**内容まとめ**:

*   **6.1 物理的常識推論**: 426動画から604問（Space 80問、Time 298問、Fundamental Physics 226問）。二択問題とMCQの組み合わせ。
*   **6.2 具現化推論**: 6つのベンチマーク、600動画から610問。
    *   **BridgeData V2**: 100問（次の即時アクション予測）
    *   **RoboVQA**: 101問（タスク完了検証、アフォーダンス）
    *   **RoboFail**: 100問（困難なアフォーダンス・タスク完了検証）
    *   **AgiBot**: 100問（次のサブタスク予測）
    *   **HoloAssist**: 100問（次のサブタスク予測）
    *   **AV（自律走行車）**: 100問（次の即時アクション予測、完了検証、アフォーダンス）

## 7. 実験 (Experiments)
**内容まとめ**:

*   **7.1 Physical AI SFT結果**:
    *   **物理的常識**: Cosmos-Reason1-56Bが60.2%で最高精度（OpenAI o1の59.9%を上回る）。7B版は+6.9%改善（47.4%→54.3%）。
    *   **具現化推論**: 7B版は+11.0%改善（50.8%→61.8%）、56B版は+10.2%改善（53.5%→63.7%）。
    *   **直観的物理学**: 既存モデルはArrow-of-TimeとObject Permanenceでランダム推測レベル。Cosmos-Reason1-7Bは平均+32.4%改善（42.1%→74.5%）。
*   **7.2 Physical AI RL結果**:
    *   物理的常識+具現化推論で平均+5.0%改善（60.7%→65.7%）。
    *   直観的物理学で+7.0%改善（74.5%→81.5%）。特にSpatial Puzzleは85.4%→94.0%。
    *   RLにより、曖昧な質問に対して全選択肢を拒否する能力が創発。

## 8. 関連研究 (Related Work)
**内容まとめ**:
- **Physical AI基盤モデル**: LLMゼロショットプランナー、VLA（Vision-Language-Action）モデル、CoT-VLAなど。
- **Vision Language Models**: Flamingo、LLaVA、InternVL、QwenVL、NVLMなど。decoder-onlyアーキテクチャが推論能力に優れることを確認。
- **推論能力を持つLLM/VLM**: OpenAI o1、DeepSeek-R1など。本研究はPhysical AIの文脈で推論能力を探求。

## 9. 結論 (Conclusion)
**内容まとめ**:
Cosmos-Reason1は、物理世界の理解と推論に特化したマルチモーダルLLMファミリーです。Physical AI向けの基盤能力を定義するオントロジーを構築し、SFTデータとベンチマークを整備しました。Physical AI RLでは、空間・時間・直観的物理学についての推論を向上させるルールベース・検証可能報酬を設計しました。
- Physical AI SFTにより、バックボーンVLMの性能が10%以上向上
- Physical AI RLにより、さらに5%以上の精度向上
- 既存モデルが苦手とする時間の矢や物体の永続性などの直観的物理学を学習可能

コードはオープンソース、モデルはオープンウェイトで公開し、Physical AIシステムの発展を促進します。
