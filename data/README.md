## English Version

# Data Directory
Directory for the data to be analyzed.

## Data Format
This script analyzes rotational motion image data in AVI format.<br>
Create a directory under data/ and place the AVI data following the naming convention [data_name]_[data_number].avi under that directory.<br>
It is recommended to binarize the rotational motion image data in advance (set the rotating object to white and the background to black).<br>
Specify the directory name as an argument when executing. (Refer to the README in scripts for details).<br>

## Config File
Create a config.ini file and define the following keys in the Setting section:

- `sample_num`: Number of data samples
- `FrameRate`: Frame rate of the recording
- `total_time`: Total recording time [s]
- `px2um_x`: Actual length of 1 pixel on the x-axis [μm/pixel]
- `px2um_y`: Actual length of 1 pixel on the y-axis [μm/pixel]

## Test Data
test_data is provided in the data/ directory. If there are any unclear points regarding procedures or settings, please refer to test_data first.


<br><br>
## 日本語バージョン

# Data directory
解析したいデータのディレクトリ

## データ形式
本スクリプトは，AVI形式の回転動画像データの解析を行う．<br>
`data/`にディレクトリを作成し，そのディレクトリ下に`[データ名]_[データ番号].avi`という命名規則に従いAVIデータを入れる．<br>
回転動画像データは，予め二値化することが推奨される（回転体は白、バックグラウンドは黒に設定）．<br>
実行の際は，引数にディレクトリ名を指定する．(詳細は`scripts`のREADMEを参考．)<br>

## configファイル
`config.ini`ファイルを作成し，`Setting`セクションに以下のキーを定義する必要がある．

- `sample_num`: データ数
- `FrameRate`: 撮影フレームレート
- `total_time`: 撮影時間 [s]
- `px2um_x`: x軸1ピクセルの実際の長さ [μm/ pixel]
- `px2um_y`: y軸1ピクセルの実際の長さ [μm/ pixel]

## test_data
`data/`に`test_data`を用意している．手順や設定について不明点がある場合は、まずは`test_data`を参考いただきたい．
