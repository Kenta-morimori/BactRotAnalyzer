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
