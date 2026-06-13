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
