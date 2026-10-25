## English version

# Scripts directory

Information about the scripts.

## 1. Setting up the environment

To install the required libraries, run the following:
```bash
pip3 install -r requirements.txt
```

## 2. Execution

Please execute under `BactRotAnalyzer`.

### 2.1. Rotation Analysis

Rotation analysis can be performed by executing the following command:<br>
Make sure `[directory_name_to_analyze]` matches a directory name in `data`.

```bash
python3 python3 scripts/02_compare_fluctuation_property/compare_fluctuation_main.py \
  --days [day1] [day2] [day3] \
  --plot-labels [label1] [label2] [label3]

# example
python3 python3 scripts/02_compare_fluctuation_property/compare_fluctuation_main.py \
  --days SJW46_temp=10 SJW46_temp=23 SJW46_temp=30 \
  --plot-labels 10℃ 23℃ 30℃
```

### 2.2. Fluctuation Analysis

To perform fluctuation analysis, add the `--fluc` argument.

```bash
python3 scripts/01_standard_analyisis/rotation_analysis_main.py --day [directory_name_to_analyze] --fluc
```

### 2.3. Comparison of Fluctuation Features

```bash
python3 scripts/01_standard_analyisis/rotation_analysis_main.py --day [directory_name_to_analyze] --fluc
```

### 2.4. Repellent Response Analysis

Repellent response analysis can be performed by executing the following command:

```bash
python3 scripts/04_repellent_response_analysis.py --day [day_directory]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--day` | str | `repellent-response/20260312` | Data directory (subdirectory in `data/`) |
| `--baseline-ratio` | float | `0.5` | Baseline ratio for rise-point detection |
| `--sigma-threshold` | float | `3.0` | Standard deviation threshold for rise-point detection |
| `--min-consecutive` | int | `3` | Minimum consecutive frames for rise detection |

**Example:**

```bash
python3 scripts/04_repellent_response_analysis.py --day repellent-response/20260416_23
python3 scripts/04_repellent_response_analysis.py --day repellent-response/20260416_23 \
  --baseline-ratio 0.4 --sigma-threshold 2.5 --min-consecutive 4
```

**Output directory structure:**

Generated outputs are saved in `outputs/{day}/repellent_response/`:

- `00_time_list/`: Time list and visualization
- `01_brightness_change/`: Background intensity and rise detection results
- `00_all_rotational_analysis/`: All-time rotational analysis, including centroid coordinates, rotation center, and angular velocity
- `02_pre_rise_fluctuation/`: Pre-rise rotational and fluctuation analyses
- `03_post_rise_analysis/`: Post-rise centroid and angular velocity analysis

## 日本語版

# スクリプトディレクトリ

解析スクリプトに関する情報です。

## 1. 環境構築

必要なライブラリをインストールするには以下を実行してください:

```bash
pip3 install -r requirements.txt
```

## 2. 実行方法

`BactRotAnalyzer` ディレクトリ下で実行してください。

### 2.1. 回転解析

回転解析は以下のコマンドで実行できます。<br>
`[directory_name_to_analyze]` は `data` 内のディレクトリ名に合わせてください。

```bash
python3 python3 scripts/02_compare_fluctuation_property/compare_fluctuation_main.py \
  --days [day1] [day2] [day3] \
  --plot-labels [label1] [label2] [label3]

# 実行例
python3 python3 scripts/02_compare_fluctuation_property/compare_fluctuation_main.py \
  --days SJW46_temp=10 SJW46_temp=23 SJW46_temp=30 \
  --plot-labels 10℃ 23℃ 30℃
```

### 2.2. 揺らぎ解析

揺らぎ解析を行うには `--fluc` オプションを追加してください。

```bash
python3 scripts/01_standard_analyisis/rotation_analysis_main.py --day [directory_name_to_analyze] --fluc
```

### 2.3. 揺らぎ特徴の比較

```bash
python3 scripts/01_standard_analyisis/rotation_analysis_main.py --day [directory_name_to_analyze] --fluc
```

### 2.4. 忌避応答解析

忌避応答解析は以下のコマンドで実行できます。

標準解析と忌避応答解析は `outputs/{day}/raw_centroid/{AVI名}.csv` の原フレーム番号付き重心を共用します。標準解析の `centroid_coordinate.csv` は中心相対座標として引き続き保存されます。

```bash
python3 scripts/04_repellent_response_analysis.py --day [day_directory]
```

**パラメータ：**

| パラメータ | 型 | デフォルト値 | 説明 |
|-----------|-----|--------|------|
| `--day` | str | `repellent-response/20260312` | データディレクトリ（`data/` 配下のサブディレクトリ） |
| `--baseline-ratio` | float | `0.5` | ベースライン比率（rise point検出用） |
| `--sigma-threshold` | float | `3.0` | 標準偏差の閾値（rise point検出用） |
| `--min-consecutive` | int | `3` | rise検出の最小連続フレーム数 |

**実行例：**

```bash
python3 scripts/04_repellent_response_analysis.py --day repellent-response/20260416_23
python3 scripts/04_repellent_response_analysis.py --day repellent-response/20260416_23 \
  --baseline-ratio 0.4 --sigma-threshold 2.5 --min-consecutive 4
```

**出力ディレクトリ構成：**

生成された出力は `outputs/{day}/repellent_response/` に保存されます。

- `00_time_list/`: 時刻リストと可視化
- `01_brightness_change/`: 背景輝度と rise検出結果
- `00_all_rotational_analysis/`: 全時間の回転解析（重心座標、回転中心、角速度を含む）
- `02_pre_rise_fluctuation/`: pre-rise の回転解析と揺らぎ解析
- `03_post_rise_analysis/`: post-rise の重心座標と角速度解析


## 忌避応答解析：V2を既定に使用

```bash
python3 scripts/04_repellent_response_analysis.py --day repellent-response/23
# 全保存フレームを解析する場合（設定済み範囲をこの実行だけ上書き）
python3 scripts/04_repellent_response_analysis.py --day repellent-response/23 --all-frames
```

V2はドリフトと停止を考慮した符号参照モデル。CWが正、CCWが負、判定不能はNaN。
角速度は中央0.2秒窓を約0.025秒間隔で評価する点系列で、出力時刻と同じ長さ。
参照中心は採用した元フレームにのみ保存し、中心CSVからの補間はしない。
既定値は`utils/param.py`の`center_estimator="v2"`と`stop_measurement_enabled=True`。
CLI指定を優先し、従来の採否付き楕円は`--center-estimator validated`、旧中心方針は`--center-estimator legacy`で実行できる。
`--no-stop-measurement`は停止中測定だけを無効化する。停止測定はV2使用時に限り実行する。

### 回転中心と停止候補の標準出力

`outputs/{day}/repellent_response/rotation_center/`へ次を保存する。

- `settings.json`: 手法・閾値・単位・時刻対応・符号規約。
- `summary.csv`: 試料ごとのV2判定数、停止候補・測定数、欠測理由別件数。
- `noXX/timeseries.csv`: 評価時刻と原フレーム、重心、V2中心・符号・角速度、停止候補、停止用中心・角速度、採否理由。
- `noXX/stop_episodes.csv`: 停止候補の開始終了と前後参照、補間可否、測定数。
- `noXX/overview.png`: 重心・中心のx/yとXY軌跡、角速度、状態。
- `noXX/diagnostics.png`: 運動量、方向一貫性、参照支持、符号一致、欠測理由。
- `noXX/episodes/stop_NNN.png`: 停止候補と前後の回転参照・補間・角速度の拡大図。

青は回転支持されたV2参照、橙は停止測定の補間参照、灰色帯は記録欠損。
停止候補は実際の停止の確定ではなく、V2欠測全体を停止とは扱わない。
停止中の`stop_angular_velocity_rad_s`は別系列であり、主角速度・CW率・切替集計には混ぜない。
前後参照不足、記録欠損、長時間停止、中心付近などは欠測を維持する。
`omega_noise_scale_rad_s`は近似ノイズ指標で、校正済みの信頼区間ではない。
実測値には中心・回転方向の独立した真値がない。評価と既知の限界は[Issue #44](https://github.com/Kenta-morimori/BactRotAnalyzer/issues/44)へ記録する。

生成物はローカル出力とし、Gitへ追加しない。入力データと`raw_centroid`は保持する。

### 最終参照中心・停止角速度と実行別出力

忌避応答解析は実行ごとに
`outputs/[suffix/]<day>/repellent_response/runs/<configured|all_frames>/<日本時間の実行ID>/`
へ保存します。`--all-frames` と通常解析は別領域になり、既存成果物を削除・移動しません。
`run.json` に成功／失敗と原フレーム範囲を記録します。
V2では最終参照中心、低信頼値を含む停止角速度、品質・理由の追加列と図を保存します。
主角速度・CW率・切替数・揺らぎ解析の入力は従来の主系列を維持します。
[列・理由・図・検証の詳細](../docs/analysis/repellent-reference-output.md)を参照してください。

### 重心の画像診断・手法比較

`05_centroid_diagnostics.py` は原画像・マスク・生重心の診断、
`06_centroid_comparison.py` は合意済み基準での分離・対象選択の比較と採用判断に使用します。
`07_centroid_review.py` は元TIFF上に候補マスク・保存済み重心を重ねたローカル比較動画を出力します。
`08_registered_centroid_review.py` は保存済み二値TIFFのフレームラベルと旧AVIを照合し、
切り出し位置を画像から推定・検証して、旧AVIと10倍拡大後の二値化を同座標で比較します。
実験結果は `outputs/centroid_evaluation/` に実行ごとに保存します。通常解析への採用候補はなく、既存解析・キャッシュは保持します。
[評価条件・実行手順・採用判断](../docs/centroid-evaluation.md)を参照してください。


### 元TIFFの1点指定・自動ROI重心抽出

```bash
python3 scripts/tiff_roi_centroid.py --input data/example/tiff_frames \
  --output outputs/tiff_centroid/run01
```

最初に対象内をクリックし、自動ROIを確認します。元TIFFを10倍線形補間してから固定閾値で抽出し、
結合・曖昧・ROI端は欠測として、分離後の再捕捉を試みます。記録ギャップ後は再指定します。
重心CSVと重ね合わせ動画を実行別に保存します。通常回転解析への接続は含みません。
[操作・設定・欠測規則・検証範囲](../docs/tiff-roi-centroid.md)を参照してください。
