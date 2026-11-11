## English version

# Scripts directory

Information about the scripts.

Analysis CLIs require committed, clean Git sources and save `manifest.json` / `run.log` under `outputs/_runs/`.
See the [development and execution guide](../docs/development-environment.md).

## 1. Setting up the environment

To install the required libraries, run the following:
```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
source .venv/bin/activate
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

解析CLIは未コミット変更がある場合に停止し、実行時のブランチ・commit・入力hash・環境・終了状態を
`outputs/_runs/` に保存します。[環境構築と実行履歴](../docs/development-environment.md)を参照してください。

## 1. 環境構築

必要なライブラリをインストールするには以下を実行してください:

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
source .venv/bin/activate
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

標準解析と忌避応答解析は `outputs/{day}/raw_centroid/<method>/{AVI名}.csv` の原フレーム番号付き重心を共用します。標準解析の `centroid_coordinate.csv` は中心相対座標として引き続き保存されます。

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


## 忌避応答解析の現在の手法・出力

既定は局所楕円中心推定＋3点平均による暫定補完です。重心と角速度は平滑化せず、CCW正で保存します。
試料ごとのフォルダ・実行IDは作らず、フェーズごとに結果を集約します。背景上昇前は揺らぎ・FFT、後は切替診断を維持します。

[実行・入力manifest・出力一覧](../docs/analysis/repellent-workflow.md) と [重要な設計判断（ADR）](../docs/adr/README.md) を参照してください。

### 重心の画像診断・手法比較

`05_centroid_diagnostics.py` は原画像・マスク・生重心の診断、
`06_centroid_comparison.py` は合意済み基準での分離・対象選択の比較と採用判断に使用します。
`07_centroid_review.py` は元TIFF上に候補マスク・保存済み重心を重ねたローカル比較動画を出力します。
`08_registered_centroid_review.py` は保存済み二値TIFFのフレームラベルと旧AVIを照合し、
切り出し位置を画像から推定・検証して、旧AVIと10倍拡大後の二値化を同座標で比較します。
実験結果は `outputs/centroid_evaluation/` に実行ごとに保存します。通常解析への採用候補はなく、既存解析・キャッシュは保持します。
[評価条件・実行手順・採用判断](../docs/centroid-evaluation.md)を参照してください。
