# 生TIFF・AVIから標準解析へ接続する

[入力契約のADR](../adr/0005-image-input-settings.md)、[Mean CSVと上昇点検出のADR](../adr/0006-provided-brightness-rise.md)、[標準解析・出力](repellent-workflow.md)を参照する。
config.iniは任意。設定JSONの草案作成、対象設定、解析実行の順に進める。
ユーザーが作成した背景Mean CSVを読み、[持続上昇検出](background-rise.md)で上昇点を検出する。背景の画像抽出・補正は追加しない。
解析CLIには[コミット済みのclean Git状態と実行履歴](../development-environment.md)の契約を適用する。
設定エディタの`--init`と`--configure`は解析結果を作らない。設定JSONはGit管理外へ保存する。

## 既定のJSONで簡単に実行する

JSONの既定位置は`data/<day>/repellent_input.json`。
初回作成では入力源とその校正を指定する。対象・閾値・ROIは対話で確認する。

```bash
python scripts/prepare_repellent_input.py --init --day repellent-response/20261027_23 --source tiff --px2um 0.2 0.2
# 撮影順を確認してから --confirm-order を付ける
python scripts/prepare_repellent_input.py --configure --day repellent-response/20261027_23
python scripts/04_repellent_response_analysis.py --day repellent-response/20261027_23
```

既定JSONが作成済みならinitは不要。背景CSVの後日追加も`--attach-brightness --day <day>`だけで登録できる。
実行時のJSONパスと入力源は省略でき、JSONのsourceを使う。明示JSONは既定より優先し、CLI入力源も指定された場合は一致を検査する。
既定JSONがなければ従来のAVI入口を維持する。不正なJSONや未確認の設定を無視して別方式へ切り替えない。
再計算manifest・結果bundleを明示した場合は既定JSONを読まない。[ADR 0007](../adr/0007-default-image-settings.md)を参照する。
以下の明示パスによる操作も、方式比較・複数設定の保存用として継続利用できる。

## TIFFの設定と実行

リポジトリ直下で実行する。校正は**入力源の元画素**に対して指定する。
今回のTIFFは0.2 μm/px、既存AVIは0.02 μm/pxであり、取り違えない。

```bash
python3 scripts/prepare_repellent_input.py --init \
  --day repellent-response/23 --source tiff --px2um 0.2 0.2 \
  --settings outputs/input_settings/23-tiff.json

python3 scripts/prepare_repellent_input.py --init \
  --day repellent-response/20261027_23 --source tiff --px2um 0.2 0.2 \
  --settings outputs/input_settings/20261027_23-tiff.json
```

`--day`に`outputs/`を付けない。別のデータ領域を使う場合は`--data-root /path/to/data`を指定する。
既存のconfig.iniがあれば撮影順と解析範囲を継承し、元configを書き換えない。
configがなければフォルダ名順の番号案を表示し、`order_confirmed=false`で保存する。
画像や時刻タグから対象・閾値・正しい試料番号・物理校正を自動確定しない。
既存設定ファイルを`--init`で上書きすることもない。
新規草案は、同名の`brightness_data/<撮影名>.csv`があれば自動検出用の背景入力として登録する。

すでに作成済みのJSONへ、後から追加したMean CSVを登録する場合は次を実行する。
番号・対象・ROI・閾値の既存設定は保持する。異なる背景設定がある場合は停止し、勝手に置換しない。

```bash
python3 scripts/prepare_repellent_input.py --attach-brightness \
  --settings outputs/input_settings/20261027_23-tiff.json
```

既定ではTIFF撮影フォルダと同じデータセットの`brightness_data/`を使う。
別の場所なら`--brightness-root /path/to/brightness_data`、一部だけなら`--samples 1 3`を指定する。

番号と撮影名の一覧を確認・必要なら編集してから対象を設定する。

```bash
python3 scripts/prepare_repellent_input.py --configure \
  --settings outputs/input_settings/23-tiff.json

# フォルダ名順の番号案を確認済みの場合だけ --confirm-order を付ける
python3 scripts/prepare_repellent_input.py --configure --confirm-order \
  --settings outputs/input_settings/20261027_23-tiff.json
```

TIFFのconfigureはローカルHTML画面を開く。端末にもURLを表示する。
[自動候補と一括承認のADR 0008](../adr/0008-batch-tiff-review.md)を参照する。

1. 「全撮影の自動設定・動画作成」を押す。保存済みの閾値・ROI・対象指定を優先し、未設定の閾値は2000、対象は画像中央に最も近い領域を候補にする。ROI未設定なら既存の冒頭区間の推定を使う。
2. 保存された横並び動画を確認する。左は原TIFF、右は10倍線形拡大後の二値マスク。同じ固定ROI、対象指定点、採用領域・重心を表示する。全時間シーク、再生速度変更、元フレーム・実時刻、欠測・再捕捉・記録境界へのジャンプを使える。
3. 問題のあるNo.だけ「対象・閾値・ROIを修正する」を開いて編集し、そのNo.だけ再生成する。問題分を保留し、「解析対象○件・保留No.」を確認して「まとめて承認して保存」を押す。

動画と候補は `outputs/preprocessing_review/<day>/<review_id>/noXX/` に保存する。
`review.mp4`、`centroids.csv`、`candidate.json`、`presentation.json`を保持し、レビュー単位の`index.html`はローカルで動画を閲覧できる。
全保存TIFFを間引かず50 fpsで収録する。動画の再生時間は撮影時刻ではなく、元フレームとTIFF時刻は対応表と画面表示で確認する。
確認動画・候補・以前の動画は終了後や承認後も残し、自動削除しない。設定変更時は新しい保存先へ該当No.だけ生成する。

仮処理は入力・実装・環境を記録するためclean Gitで開始する。処理中も画面は操作でき、中断・終了時は完了済み成果物と修正要求を保持する。
再起動後の作成操作は一致する完了済み結果を再利用し、未完了分を処理する。処理未完了・自動対象/ROI未決定の撮影は承認しない。
対象やROIを決められない撮影も全視野の確認動画を作るが、全視野表示を推定ROIとして採用しない。
欠測・再捕捉・ROI端接触は確認箇所として表示し、それだけで撮影全体を承認不可にしない。

候補・動画は自動保存するが、解析用JSONの更新は一括承認時だけ行う。No.別の必須確認チェックや事前の手動対象指定は不要。
撮影順が未確認の場合だけ一覧の順序を確認する。承認時は他画面による設定変更、元TIFF・動画・測定・実装の不一致を検査する。
一括承認したNo.をJSONへ明示保存し、番号指定なしの解析はその対象を使う。保留・未承認No.を明示指定すると再生成・承認を案内する。
旧JSONに一括承認情報がない場合は従来どおり全指定試料の確認状態を要求する。
背景だけの変更は画像条件の承認を無効にしないが、対象・閾値・ROI・校正・範囲や元TIFFが変わった場合は再生成・再承認する。

終了は「設定画面を終了」。未再生成の修正や処理中の前処理があれば確認し、自動承認しない。解析も自動開始しない。Ctrl-Cも使える。
終了画面と端末に保存済みの解析対象・保留No.と実行可能なコマンドを表示する。
HTMLは127.0.0.1限定で、動画のシークにもローカルトークンを要求し、外部アセットや画像送信を使わない。
従来の撮影ごとのOpenCV設定は`--opencv`で利用できる。AVI設定の入口は維持する。

### 報告書と同じ条件を使う

TIFF補助例の条件は、10倍線形拡大後の固定閾値2000、対象指定後の自動固定ROI、
採用領域の面積重心と追跡である。自動処理はROI案の作成と対象追跡などを指し、閾値の自動最適化ではない。
HTMLは閾値が未設定の場合だけ2000を提示し、読込時に初期候補かJSONの保存値かを表示する。
保存済みの閾値・ROIは変更せず再利用する。ROI未設定時の全画像表示は、自動ROIの決定結果ではない。
通常は中央対象と自動ROIの候補を一括生成し、動画確認後に承認する。候補が違う撮影だけ対象点やROIを修正する。
報告書の特定撮影のROI座標を他の撮影へコピーしない。

一方、保存重心を固定した中心手法比較の再現には、[measurement manifestによる再計算](repellent-workflow.md#同じ入力から再計算する)を使う。
元フレーム・時刻・校正・重心の定義・背景・元フレームでの上昇境界を同じ入力から復元し、別suffixへ出力する。
この再現では保存境界を使い、今回の新しいTIFF入力ではMean CSVから持続上昇検出を使う。
画像から新たに抽出した面積重心へ差し替えると、保存済み輪郭平均重心を使った報告書の再現にはならない。
実測のmanifest・設定・照合結果はローカルに保持する。

```bash
python3 scripts/04_repellent_response_analysis.py \
  --day repellent-response/23 --input-source tiff \
  --input-settings outputs/input_settings/23-tiff.json

python3 scripts/04_repellent_response_analysis.py \
  --day repellent-response/20261027_23 --input-source tiff \
  --input-settings outputs/input_settings/20261027_23-tiff.json
```

一括承認情報があれば承認対象だけを解析する。旧JSONでは未確認の撮影があれば抽出を開始しない。一部を先に解析するには`--input-samples 1 3`を指定する。
元の番号を維持し、選択順で再採番しない。`--all-frames`で設定範囲を明示的に解除できる。
`23`のNo.14はconfig由来の2049–16384を継承する。元の手動上昇40秒は背景解析側の参考設定として元configに保持するが、新入口へ自動適用しない。
背景を未提供なら全時間の重心・中心・角速度までを計算し、前後解析は未実行とする。
Mean CSVを登録した撮影では自動検出した境界を使い、上昇前の揺らぎ・FFT、後の切替診断を実行する。
既定の持続検出と診断、明示的な旧方式、画像承認の互換移行は[背景上昇の手順](background-rise.md)を参照する。

既定出力先は`outputs/reanalysis_tiff/<day>/repellent_response/`。
`--output-suffix <name>`で保存先を明示できる。既存の`outputs/<day>/`やAVIの結果を上書きしない。
フェーズ別集約、ラベル有無、details、index.htmlとファイル一覧は既存の契約を使う。

## 同じデータのAVIから再解析する

```bash
python3 scripts/prepare_repellent_input.py --init \
  --day repellent-response/23 --source avi --px2um 0.02 0.02 \
  --settings outputs/input_settings/23-avi.json

python3 scripts/04_repellent_response_analysis.py \
  --day repellent-response/23 --input-source avi \
  --input-settings outputs/input_settings/23-avi.json
```

AVIの面積重心抽出は既存の閾値120・最大外輪郭の連結領域を使い、対象追跡法をTIFFと同一視しない。
TIFF元時刻を使い、AVI内順序番号と元TIFF接尾辞の対応をCSVに残す。
撮影名と観測数が一致しなければ中断する。対応と対象同一性を画像内容から検証したとは扱わない。
既定出力先は`outputs/reanalysis_avi/<day>/repellent_response/`。新経路の専用キャッシュから再抽出するため、既存AVIキャッシュを手動削除する必要はない。
`--force-extract`でその設定の重心を取り直せる。AVIの明示入力でも同じMean CSV・上昇点検出を利用できる。

## 設定と外部背景の契約

| 設定 | 役割 |
|---|---|
| config.ini | 新しい画像入力入口では任意。存在すれば草案作成時に撮影順・解析範囲・AVI校正を参照する |
| 共通JSON | 必須。入力源、元試料番号、対象・閾値・ROI、校正、範囲、背景入力を保存する |
| `--input-source tiff` / `avi` | 任意。省略時はJSONのsourceを使い、指定時は一致を検査する |

JSON指定後の実行は元config.iniを必要としない。設定なしで対象や物理校正を確定する機能ではないため、JSON草案の作成・確認は必要である。

JSONの概略（例示値であり、実測の対象点ではない）:

```json
{
  "version": 1,
  "day": "repellent-response/example",
  "source": "tiff",
  "order_confirmed": true,
  "background_policy": "external_only",
  "samples": [{
    "sample_no": 3,
    "input_id": "recording-name",
    "tiff_directory": "../../data/repellent-response/example/tiff_data/recording-name",
    "px2um_x": 0.2, "px2um_y": 0.2,
    "analysis_frame_range": null,
    "target_confirmed": true,
    "threshold": 2300,
    "roi": [10, 10, 44, 44],
    "seeds": {"1": [30, 32], "100": [31, 32]},
    "background": null
  }]
}
```

ファイルパスはJSONからの相対パスまたは絶対パス。草案は絶対パスを保存する。
`seeds`は1始まりの元フレーム番号をキーとし、再指定位置で対象の連続性を区切る。
再指定の追加、ROIや閾値変更はJSONを編集するか`--configure`で行う。対象同一性が不明な区間は欠測を維持する。
`analysis_frame_range`は元フレームの両端を含む範囲、`null`は全保存行。記録gapや欠測を詰めたり架空の時刻を補ったりしない。

今回の`brightness_data`形式は対象sampleの`background`を次のように指定する。

```json
{"csv": "../../data/repellent-response/example/brightness_data/recording-name.csv", "format": "imagej_mean", "rise_detection": "auto"}
```

必須列は`[inch],Mean`。CSVの全行数が保存TIFF枚数と一致しなければ停止する。
Mean行は元スタックの保存順で対応付け、`[inch]`は時刻に使わない。元フレーム・時刻はTIFFから取得する。
撮影名と行数の整合を検査するが、CSVにフレームIDがないため、行同一性を独立に検証したとは扱わない。
保存順に作成したCSVを使うこと。

すでに元フレーム・時刻を持つ背景CSVも利用できる。手動で境界を指定する例:

```json
{"csv": "user-background.csv", "rise_source_frame": 1200}
```

この形式の必須列は`source_frame_1based,time_sec,background_intensity`。
解析対象範囲または全記録の行を含め、**元フレームと時刻を重心行に完全一致**させる。
時刻は最初の有効TIFF時刻を0秒とする。欠測時刻は空欄のまま保存し、CSV書き出しは17桁精度を推奨する。
`rise_source_frame`を指定すると自動検出より優先する。元フレーム/時刻形式でも`rise_detection: "auto"`を追加すれば自動検出できる。
自動検出・手動のどちらの境界も、刺激到達や物理的反転の確定ではない。
背景未提供・検出なし・検出行の元時刻や輝度が欠測の場合は前後指標を未実行とする。元フレーム/時刻形式で境界・auto指定の両方を省略した場合も同様。
検出方法・状態は`rise_summary.csv`、パラメータと診断値は`settings.json`の`image_input.rise_results`に記録する。
背景の再解析だけなら抽出設定を変えず、同じコマンドを再実行できる。
既存config.iniの手動境界は、この外部入力へ自動転記しない。

`details`、`samples_per_page`、`zoom_ranges`は[標準manifest](repellent-workflow.md)と同じ意味でトップレベルへ追加できる。

## 保存・再現・限界

標準入力manifestは`outputs/[suffix/]<day>/measurement_input/`、重心CSVとメタデータは同じルートの`raw_centroid/<method>/`へ保存する。
入力・抽出設定・範囲・コード・環境のハッシュをファイル名に使い、試料別・実行ID別のフォルダを作らない。
異なる入力源・手法・校正・閾値・ROI・対象点のキャッシュは混用せず、既存AVIキャッシュに書き戻さない。
改変・破損したキャッシュは再抽出する。古い設定や元画像を参照するmanifestは、変更を検出すると再生成を要求する。
生成済みmanifestを`--measurement-input <path>`で指定すれば中心以降だけを再計算できる。

TIFFの抽出規則・接触時の限界は[独立抽出の仕様](../tiff-roi-centroid.md)を参照する。
接触・曖昧・ROI端は理由付き欠測、ギャップ後の未指定も欠測とし、原重心を補完しない。
再捕捉や見えない間の入れ替わりの正確さは保証しない。抽出結果と後半の対象同一性は別途確認する。
標準解析へ接続できることは、任意の実測での精度合格や新しい反転判定基準の採用を意味しない。
実測の設定・画像・座標・時系列・図はローカルのみで保存する。
