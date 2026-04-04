commitするときは適切な粒度でconventional commitでお願いします。prefixのあとのカッコはつけないでください。

clean git worktree 制約を回避するために、別ディレクトリや別 worktree に設定ファイル、実験用コード、実行用 repo を逃がすことを禁止します。

実験に必要な設定変更や実装変更は、このプロジェクト直下で行ってください。clean worktree が必要なら、勝手に別ディレクトリへ逃がさず、まずこのプロジェクトで commit するか、ユーザーに確認してください。
