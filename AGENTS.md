# 開発共通運用

## Issueと着手判断

- Issueのタイトル・本文・重要コメント・引き継ぎプロンプトは基本日本語。識別子・コマンド・必要な引用・既存履歴は原表記を保つ。
- 初回は全体の入口（#58）、分野親、直接の親、対象Issueの最新本文・重要コメント、直接の依存、関連PRとDevelopment、[Project 12](https://github.com/users/Kenta-morimori/projects/12)を確認する。再開時は前回以降の差分と現状を中心に確認し、毎回全履歴を読み直さない。
- Issueには全体の入口・直接の親・目的・範囲・対象外・受入条件・必要な入力・依存・優先度と理由・期限の状態を記載する。直接の親は一つとし、階層は目的と責任、依存は着手や完了を制約する条件として分ける。親の全完了を独立タスクの着手条件にしない。
- 新規作成前に既存Issueと同種の親を確認する。重複は既存を再利用し、一部重複は役割・関連・依存を整理して同じ修正を重複実装しない。過去の完了・検証・複数PRの履歴は保持する。
- 優先順位はユーザー指定を最優先とし、その範囲で着手可能性、重要な依存の解除、期限、研究への影響を判断する。過剰な数値スコア化は不要。進行中の担当を確認し、同じタスクへの重複着手を避ける。

## 期限とKanban

- 期限は「確定・暫定・未定」を区別し、日本時間の具体的な日付と根拠を本文に記す。ユーザー指定を優先し、指定がなければ関連目標と作業規模から暫定日を置く。根拠がなければ未定理由と次の見直し日を記し、無理な日付を作らない。
- 日付がある場合は本文とProject 12のTarget dateをそろえる。未定に変更した場合は古いTarget dateを残さない。親の全体目標を全ての子にコピーしない。着手時に担当・容量・依存から妥当性を見直す。
- 期限変更時は旧日・新日（または未定）・理由・影響を重要コメントに残す。遅延時は親と依存先への影響を確認し、日付を一律にずらさない。期日接近だけで品質条件や受入条件を黙って緩めない。
- 研究側#62の2026-10-31目標への影響を考慮し、開発環境整備全体を研究解析の必須依存にしない。この共通運用だけを根拠に通知・監視・期限リマインダーを新設しない。

実装IssueのStatusは次の条件で扱う。待機理由・不足情報・解除条件・進められる独立作業は本文へ記録し、新しい待機列を増やさない。

| Status | 条件 |
| --- | --- |
| Backlog | 未整理、優先待ち、または条件不足。 |
| Ready | 目的・範囲・受入条件・必要な入力と依存がそろい、着手できる。 |
| In progress | 担当が着手し、作業領域と変更範囲を確認した。 |
| In review | レビュー可能なPRがあり、検証結果と未解決事項が分かる。 |
| Done | 実際にマージ済みで、受入条件・必要な確認・PR対応を満たす。 |

管理専用の親は子の進捗と親自身の完了条件で判断する。子の一部が作業中なら進行状況を反映し、統合確認待ちならその理由を本文に記す。子一件の完了だけで親をDoneやクローズにせず、継続管理する索引は開いたまま維持する。管理親に空PRやマージを要求しない。

## 作業領域・PR・完了

- 実装Issueごとに原則1 PR。既存PR・ブランチを調べ、修正は同じPRへ追加する。並列タスクは専用ブランチ（既定は`codex/`）と作業領域を分け、共有チェックアウトの無関係な変更・実験成果物を持ち込まない。
- 共通ファイルの編集担当は一つにする。他担当に必要な変更は該当Issueへ要点を残し、変更範囲や受け渡しの影響を整理する。相互のPRに同じ修正を含めない。
- PRは対象IssueのDevelopmentで実際の紐付けを確認する。本文の単なる参照だけでは確認済みにしない。baseと既定ブランチの違いにも注意する。
- 範囲内の可逆な実装判断・修正・必要な検証は自律的に進める。実装自己テストを独立レビュー完了として扱わず、レビュー・CIは対象のbaseと最新head SHAを明確にする。head更新後は前回指摘・新規差分・関連影響を確認し、古いheadの確認を最新と呼ばない。
- マージは既存のユーザー指示・権限・承認条件に従い、この運用で承認フローを増やさない。未マージは完了扱いにしない。実際のマージ後に受入条件・必要な確認・PR対応、Issue・親の進捗・Kanbanをそろえる。
- マージ後は現在の優先度・期限・依存・進行中の担当から着手可能な次Issueと理由、そのまま投入できる日本語プロンプトを提示する。プロンプトには対象・親・全体のリンク、目的・範囲・受入条件への参照、関連PR、必要な前提・制約を含める。次タスクの自動開始・新チャット作成・他チャットへの送信は別途指示がある場合だけ行う。

## 判断・記録・skillへの入口

- 重要な目的・受入条件・優先順位の変更、科学的な測定や解釈の未確定事項、重要なトレードオフ、回復困難な操作や既存の権限・費用・公開範囲を越える行為、既存規則で明示された承認では、決めること・推奨案・代替案と影響をまとめてユーザーに確認する。通常の実装詳細や承認済み作業を聞き直さない。回答待ちは時間経過で承認に置き換えず、独立作業を進める。
- 報告は重要な変更・影響・検証の要点・未解決事項を簡潔に述べる。毎回の報告書・作業日誌は作らない。現在の合意はIssue本文、重要な決定・方針変更・重大なブロッカー・検証の限界・マージ後の引き継ぎは対象Issueコメント、実装と検証の根拠はPRへ残す。親へは影響がある場合のリンクと要点だけを記録する。
- 受け渡しには対象Issueと親、関連PR、作業領域/ブランチ、base/最新head SHA、変更範囲、検証結果、未解決事項、依存と期限を使い、初回と変更時に必要な差分を渡す。
- 共通原則は本ファイル、詳しい手順は各skillに置く。導入済みのレビュー入口は[同梱bactrot-code-review](.agents/skills/bactrot-code-review/SKILL.md)。実装・引き継ぎskillを使う場合は、その環境で導入済みと確認できた入口を読む。未導入の兄弟skillや将来のパスを必須の実行依存にせず、本規約と担当Issueで単独検証・マージできる構成にする。

## Code Review Rules

### PR artifact cleanup

- Before requesting review and before merging, inspect the complete PR diff for temporary debugging or comparison helpers, one-off scripts, work logs, hard-coded local paths, and files named for the PR or issue rather than their continuing purpose.
- Remove artifacts needed only to prepare or review the PR. Keep reusable feature code, regression tests, specifications, operating instructions, and validation evidence; a PR or issue reference alone is not a reason to delete them.
- Organize permanent documentation by feature or concept so it remains understandable without the PR discussion. Keep development chronology and review receipts in the issue/PR or local notes, rather than production documentation or runtime code.
- Remove unused code, options, dependencies, and obsolete references; update links and check that the remaining documentation and commands refer to existing files. Summarize this cleanup in the PR checklist.
- Do not include or delete unrelated work or existing experimental artifacts during cleanup. Preserve the experimental data boundary below.

### GitHub Codex review requests

- Request a GitHub Codex review manually at most once per PR by default. Check the PR's existing Codex review history before posting `@codex review`; a new head does not reset the count.
- For any additional GitHub Codex review, explain the reason, the previously reviewed and current head SHAs, the findings addressed, and the current checks, then obtain explicit approval before posting another request. Local review and CI may continue without another GitHub request.

### Frame and time identity

- In repellent response analysis, keep each observation tied to its original frame and timestamp. Missing coordinates, invalid centers, and timestamp gaps must remain unavailable; do not compact rows, carry values across gaps, or calculate a direction or switching event across them. Preserve explicit source-frame mapping when saving or splitting a series.

### Primary rotation and stop measurements

- For V2 repellent-response output, treat the V2 sign, reference center, and angular velocity as the primary rotating series. Stop-candidate center and angular velocity are separate measurements: they must not fill missing primary values or enter primary CW rate and switching counts. If the required references or lever arm are unavailable, retain a missing stop measurement with its reason.
- AVI centroid coordinates use a downward-positive y axis. In this image space, a visually clockwise orbit has positive angular change; repellent-response output reports CW as positive for V2, validated, and legacy estimators.

### Experimental data boundary

- Do not commit or post measured coordinates, raw images or videos, reproducible measured time series, or generated plots derived from them. Keep those artifacts local. Use synthetic data and aggregate, non-reconstructable summaries for tests and public issue or PR evidence.
