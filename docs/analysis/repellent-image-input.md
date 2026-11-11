# 生TIFF・AVIから標準解析へ接続する

[入力契約のADR](../adr/0005-image-input-settings.md)、[Mean CSVと上昇点検出のADR](../adr/0006-provided-brightness-rise.md)、[標準解析・出力](repellent-workflow.md)を参照する。
config.iniは任意。設定JSONの草案作成、対象設定、解析実行の順に進める。
ユーザーが作成した背景Mean CSVを読み、従来の自動処理で上昇点を検出する。背景の画像抽出・補正は追加しない。
解析CLIには[コミット済みのclean Git状態と実行履歴](../development-environment.md)の契約を適用する。
設定エディタの`--init`と`--configure`は解析結果を作らない。設定JSONはGit管理外へ保存する。

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

OpenCVのデスクトップウィンドウが必要。対象内をクリックし、閾値スライダーのマスクを確認してEnterで確定する。
候補値2000は全撮影の採用値ではない。閾値は撮影ごとに固定する。
続いて自動固定ROIをEnterで確認する。Rで矩形を修正、Escで中断する。
元フレーム・時刻の記録境界後にも対象を明示的に指定する。ROIは記録全体で固定する。
`--samples 1 3`で一部だけ設定でき、確認済みの各撮影設定を順に保存する。
閾値・ROI・対象点は元TIFF画素座標でJSONに保存し、以後の再実行に使う。
対象指定が不正・ROI外・前景でない場合は保存せず中断する。

```bash
python3 scripts/04_repellent_response_analysis.py \
  --day repellent-response/23 --input-source tiff \
  --input-settings outputs/input_settings/23-tiff.json

python3 scripts/04_repellent_response_analysis.py \
  --day repellent-response/20261027_23 --input-source tiff \
  --input-settings outputs/input_settings/20261027_23-tiff.json
```

未確認の撮影があれば抽出を開始しない。一部を先に解析するには`--input-samples 1 3`を指定する。
元の番号を維持し、選択順で再採番しない。`--all-frames`で設定範囲を明示的に解除できる。
`23`のNo.14はconfig由来の2049–16384を継承する。元の手動上昇40秒は背景解析側の参考設定として元configに保持するが、新入口へ自動適用しない。
背景を未提供なら全時間の重心・中心・角速度までを計算し、前後解析は未実行とする。
Mean CSVを登録した撮影では自動検出した境界を使い、上昇前の揺らぎ・FFT、後の切替診断を実行する。
`--baseline-ratio`、`--sigma-threshold`、`--min-consecutive`はこの入口でも従来と同じ検出器へ渡す。

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
| `--input-source tiff` / `avi` | 実行する入力源を明示し、JSONの`source`との一致を検査する |

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
