# Scripts directory
Information about the scripts.

## 1. Setting up the environment
To install the required libraries, run the following:
```bash
pip3 install -r requirements.txt
```

## 2. Execution
Navigate to the `scripts` directory. Execution is done via `..._main.py`.

### 2.1. Rotation Analysis
Rotation analysis can be performed by executing the following command:<br>
Make sure `[directory_name_to_analyze]` matches a directory name in `data`.

```bash
python3 rotation_analysis_main.py [directory_name_to_analyze]
```

### 2.2. Fluctuation Analysis
```bash
python3 fluctuation_analysis_main.py [directory_name_to_analyze]
```

## 3. About the files
Description of the Python files used.

### 3.1. main.py
The main file for the user to execute. By changing the arguments, the type of analysis can be modified.

### 3.2. functions
A directory containing Python files used for analysis. The functionality of each file is described below:
- `fluctuation_analysis.py`: Performs fluctuation analysis.
- `frequency_analysis.py`: Performs frequency analysis.
- `get_angular_velocity.py`: Scripts related to time series angle and angular velocity. Includes obtaining, plotting, etc., of time series angle and angular velocity.
- `get_centroid_coordinate.py`: Scripts related to centroid coordinates. Obtains, plots, and saves centroid coordinates from AVI data.
- `input_data.py`: Handles data input.
- `make_evaluate_switching.py`: Evaluates switching of rotation direction.
- `make_graph.py`: Creates graphs.
- `param.py`: Stores parameters used by many scripts.
- `save2csv.py`: Saves data in CSV format.



# Scripts directory
スクリプトに関する情報を記載

## 1. 実行環境設定
使用するライブラリのインストールのため，以下を実行．
```bash
pip3 install -r requirements.txt
```

## 2. 実行
`scripts`に移動する．実行は`..._main.py`により行う．

### 2.1. 回転解析
回転解析は，以下のように実行することで行う．<br>
`[解析したいディレクトリ名]`は，`data`のディレクトリ名と一致するようにする．

```bash
python3 ratation_analysis_main.py [解析したいディレクトリ名]
```

### 2.2. 揺らぎ解析
```bash
python3 fluctuation_analysis_main.py [解析したいディレクトリ名]
```

## 3. ファイルについて
使用するPythonファイルの説明．

### 3.1. main.py
基本的にユーザーが実行するファイル．引数を変更することで解析の種類が変更可能．

### 3.2. functions
解析にしようするPythonファイルをまとめたディレクトリ．以下ファイル別に機能を説明する．
- `fluctuation_analysis.py`: 揺らぎ解析を行う．
- `frequency_analysis.py`: 周波数解析を行う．
- `get_angular_velocity.py`: 時系列角度・角速度に関するスクリプト．時系列角度・角速度の取得，プロット等を行う．
- `get_centroid_coordinate.py`: 重心座標に係るスクリプト．AVIデータから重心座標の取得し，プロット・保存を行う．
- `input_data.py`: データのインプットを行う．
- `make_evaluate_switching.py`: 回転方向の切り替えの評価．
- `make_gragh.py`: 図の作成を行う．
- `param.py`: 多くのスクリプトで用いられるパラメータを格納．
- `save2csv.py`: CSV形式でデータを保存する．
