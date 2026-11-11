# 開発環境と実行履歴

## 環境の準備

CIに合わせてPython 3.10を使う。`.python-version` も同じ版を指定する。
リポジトリのルートで専用の仮想環境を作成する。

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
source .venv/bin/activate
```

uvを使う場合は `uv venv --python 3.10 .venv` と
`uv pip install --python .venv/bin/python -r requirements.txt` でもよい。
科学計算ライブラリの既存の版条件は維持する。依存関係全体のlockファイルは未導入なので、
インストール時点の完全な環境固定は保証しない。実行時の全パッケージ版をmanifestへ記録する。

## 実行前の確認

`scripts/` の解析CLIは、追跡済みファイルの変更・ステージ済み変更・削除・未追跡ファイルが
一つでもあれば、解析出力の作成前にエラーで停止する。Git管理外、HEADがない場合、Git確認失敗も停止する。
Gitの除外対象であるデータ・生成物・仮想環境は判定対象外。除外済みコードを解析実装に使わない。

```bash
git status --short
git diff
# 必要なコード・設定・ADRだけを選択してcommitする。
# 実測データ・実測由来の結果はcommitしない。
python scripts/04_repellent_response_analysis.py --day repellent-response/<dataset>
```

一時変更を退避する場合も、内容を確認して対象を選ぶ。実験データをGitへ追加してcleanにしない。
dirty状態の実行を許すCLIオプションや環境変数は設けない。
`--help`、引数の構文確認、pytestによる開発中のテストは未コミットでも利用できる。
Gitゲートは解析CLIの境界に置き、下位のPython関数には置かない。
独自の解析CLIを追加するときも `utils.run_context.analysis_run` を必ず通す。

## 保存するログ

解析結果の配置は各機能の出力契約に従う。実行履歴だけを次の独立した場所に保存する。

```text
outputs/_runs/YYYY-MM-DD/HHMMSS_microseconds_random/
├── manifest.json
└── run.log
```

日付と時刻はAsia/Tokyo。秒以下と乱数を使い、同秒の再実行でも履歴を上書きしない。
開始時と終了時にmanifestの場所をコンソールへ表示する。

| 保存項目 | 内容 |
|---|---|
| Git | ブランチ、完全なcommit SHA、clean状態、detached HEADの識別 |
| コマンド | Python実行ファイル、argv、作業ディレクトリ、既定値を含む解析引数 |
| 入力 | 絶対パス、サイズ、SHA-256、見つからない入力候補・任意キャッシュ |
| 環境 | Python、OS、インストール済みパッケージ名と版 |
| 出力 | 指定した出力先または出力範囲。実際の詳細パスはrun.logと機能別manifestを参照 |
| 経過 | 開始・終了時刻、経過秒、running / complete / failed / interrupted |
| 診断 | stdout、stderr、root loggerへ伝播するログ、例外とtraceback |

通常解析ではデータセット全体と共有キャッシュ候補を列挙する。
再計算・再描画の入力manifestは、descriptor自身だけでなく参照する座標・メタデータ・
背景・TIFF・結果表もhash化する。診断CLIでは指定した診断bundleとtimelineの外部TIFFも含める。
ディレクトリsymlinkも参照先を辿り、同じ実ディレクトリは一度だけ列挙する。
入力範囲の一覧は**実行前に存在した入力候補の集合**であり、OSレベルのファイルアクセス履歴ではない。
再利用されなかったキャッシュや未選択のフレームを含む場合がある。

入力hashの計算中にファイルのサイズ・更新時刻が変われば停止する。
全動画・TIFFを読むため、大きなデータセットでは解析前のI/O時間が増える。
Git状態は入力列挙後と解析終了時にも検査し、差異があればfailedにする。
入力ファイル自体のロック・実行中の全入力再hashは行わないため、解析中にデータを書き換えない。
実行中のコード変更も避ける。終端検査までに変更を戻す操作まで検出する仕組みではない。

例外・Ctrl-Cでも履歴を保存する。強制終了・電源断はrunningのまま残ることがあるため、completeとして扱わない。
Gitゲートでの拒否と引数エラーには実行履歴を作らない。ネイティブライブラリや独立した子プロセスが
直接ファイル記述子へ出力する文字列は、Pythonのstdout/stderr捕捉では保存されないことがある。

ログも入力パスや診断値を含み得るためローカル専用とする。`outputs/_runs/` はGit管理しない。
Issue/PRには合成データによる検証か、実測を再構成できない集計だけを掲載する。
ログの自動削除は行わず、必要な履歴の退避・削除は利用者が判断する。

## 開発時の検証とADR

```bash
make test-runtime  # 一時Gitリポジトリ・合成入力で実行契約を検証
make check         # black / isort / ruff / 全pytest（CIと共通の検証内容）
make typecheck     # mypy。必要な型stubが未導入なら設定に従って取得する
```

別の環境では `make PYTHON=/path/to/python check` と指定できる。
単体テストと統合テストは解析CLIのcleanゲートとは別に実行できる。
実測解析をテスト代わりに未コミットで実行しない。

科学的手法・フレーム/時刻・単位/符号・中心補完・主指標・既定出力・実行再現性の重要な変更には、
実装と一緒に [ADR](adr/README.md) を追加し、索引と実行手順を更新する。
PRテンプレートでも確認する。本実行契約は [ADR 0004](adr/0004-execution-provenance.md) に従う。

## prj-flagella-estimationからの再利用

参照元は同プロジェクトの `src/sim_swim/core/run_context.py`、`.python-version`、
`pyproject.toml`、`.githooks/pre-commit` とCI設定である。

| 設計 | このプロジェクトでの扱い |
|---|---|
| clean Gitの必須化、ブランチ/SHAの記録 | 採用。Git確認失敗時も停止し、worktreeとdetached HEADに対応 |
| JSTの実行ディレクトリ、run.logとmanifest | 採用。解析結果の既存配置を保ち、履歴だけを独立して保存 |
| 入力・設定・環境の保存 | 採用。入力と参照先のhash、CLI引数、全パッケージ版を記録 |
| Python版と検証コマンドの統一 | 採用。既存CIのPython 3.10とMakefileで統一 |
| uv.lock、runtime/dev依存の分離 | 次の候補。既存のNumPy/OpenCV等で合成回帰を確認してから移行する |
| commit前の軽量検証hook | 次の候補。まずmake test-runtimeを使用。Git hookは各利用者が有効化する方式が適する |
| 重い処理・描画を分けるpytest marker | 次の候補。実行時間を測り、CIの待ち時間が問題になった時点で導入する |

同プロジェクト固有のクラスタ・queue・simulation階層や大規模Issue運用は導入しない。
