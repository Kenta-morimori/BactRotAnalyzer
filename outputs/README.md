## English version
# Data Directory
Directory for the analysis results.

## Format
The analysis results are saved in the outputs directory, under the corresponding directory name from the data directory.<br>

## angular_velocity
Stores the results of angular velocity analysis.<br>

- `angle_time-series.png`, `angle_time-series.csv`: Time-series angles. Defined by fitting an ellipse to the overall centroid coordinates and calculating the angle from the relationship between the center of the ellipse and each centroid coordinate. Expressed in radians.
- `angle_FFT.png`, `angle_FFT.csv`: FFT (Fast Fourier Transform) of the time-series angles. The peak (frequency at maximum amplitude) is indicated by a red line.
- `angle_FFT_peak.csv`: Peak frequency of the FFT of time-series angles. Represents the rotational speed.
- `angular-velocity_time-series.png`, `angular-velocity_time-series.csv`: Time-series angular velocity. Calculated by integrating the change in angle between each frame with the frame rate. Counterclockwise angular velocity is considered positive.
- `angular-velocity_time-series_averaged.png`: Averaged time-series angular velocity. Obtained by averaging the data over a sliding window of a specific width.
  - Only available from rotation_analysis_main.py.

## centroid_coordinate
Stores the results of centroid analysis.<br>

- `trajectory.png`: Time-series centroid coordinates.
- `x_coordinate.png`: Time-series x-coordinates of the centroid.
- `y_coordinate.png`: Time-series y-coordinates of the centroid.

## fluctuation_analysis
Stores the results of fluctuation analysis. Obtained by running fluctuation_analysis_main.py.<br>

- SD-time-series: Defined as the time-series standard deviation (SD) of fluctuations, obtained by sliding a window of a specific time width.
  - `SD-time-series_[time]s.png`: Time-series SD obtained with a time width of [time].
  - `SD-time-series_0.1s_standardized.png`: Standardized (mean 0, standard deviation 1) time-series SD. Added because the scale of time-series SD varies with the fluctuation acquisition time width.
  - `SD-time-series_all.png`, `SD-time-series_all_standardized.png`: Time-series SD for all fluctuation acquisition time widths, saved before and after standardization.
  - `SD-time-series_FFT_[time]s_standardized.png`: FFT of the standardized time-series SD.
  - `SD-time-series_FFT_all_standardized.png`: FFT of the standardized time-series SD for all fluctuation acquisition time widths.

## CSV Files
- `saved_centroid_coordinate.csv`: Stores the time-series centroid coordinates.
- `switching_value.csv`: Quantifies the switching of rotational direction. Counts the positive and negative values of the time-series angular velocity as clockwise and counterclockwise rotations, respectively. Saves the ratio.
- `switching_value_averagedAV.csv`: Quantifies the switching of rotational direction for the averaged time-series angular velocity.
  - Saved when `flag_evaluating_switching` in `/scripts/functions/param.py` is set to `True`.
  - Only available from `rotation_analysis_main.py`.


<br><br>
## 日本語バージョン

# Data directory
解析結果のディレクトリ

## 形式
`outputs`には，`data`ディレクトリに対応するディレクトリ名で解析結果が保存される．<br>

## angular_velocity
角速度解析の結果を保存．<br>
- `angle_time-series.png`, `angle_time-series.csv`: 時系列角度．全重心座標に対し楕円フィッティングをし，楕円の中心と各重心座標の位置関係から角度を定義した．ラジアンで表記している．
- `angle_FFT.png`, `angle_FFT.csv`: 時系列角度のFFT(高速フーリエ変換)．ピーク(最大のAmp時のfrequency)を赤線で示している．
-  `angle_FFT_peak.csv`: 時系列角度のFFTのピーク(frequency)．回転速度を表している．
- `angular-velocity_time-series.png`, `angular-velocity_time-series.csv`: 時系列角速度．各フレーム間の角度の変化にフレームレートを積算することで算出．反時計周りの角速度を正にした．
- `angular-velocity_time-series_averaged.png`: 平均化処理を行なった時系列角速度．特定の幅の窓関数をずらしながら平均値を算出することで平均化処理を行なった．
  - `ratation_analysis_main.py`のみで取得できる．

## centroid_coordinate
重心解析の結果を保存．<br>
- `trajectory.png`: 時系列の重心座標．
- `x_coordinate.png`: 時系列重心x座標．
- `y_coordinate.png`: 時系列重心y座標．

## fluctuation_analysis
揺らぎ解析の結果を保存．`fluctuation_analysis_main.py`を実行することで得られる．<br>
- `SD-time-series`: 揺らぎの取得方法を特定の時間幅をずらしながら取得する時系列標準偏差(SD)と定義した．
  - `SD-time-series_[time]s.png`: 設定した時間幅`[time]`で取得した時系列標準偏差
  - `SD-time-series_0.1s_standardized.png`: 時系列SDを標準化(平均0, 標準偏差1)にしたもの．揺らぎ取得時間幅により，時系列SDのスケールが異なるため追加．
  - `SD-time-series_all.png`, `SD-time-series_all_standardized.png`: 全ての揺らぎ取得時間幅での時系列SDを並べたもの．標準化前後で分けて保存している．
  - `SD-time-series_FFT_[time]s_standardized`: 標準化後時系列SDのFFT．
  - `SD-time-series_FFT_all_standardized.png`: 全ての揺らぎ取得時間幅での標準化時系列SDのFFTを並べたもの．

## csvファイル
- `saved_centroid_coordinate.csv`: 時系列重心座標を保存．
- `switching_value.csv`: 回転方向の切り替えを定量化．時系列角速度の正・負の値を時計回り・反時計回りとしてカウント．その比率を保存
- `switching_value_averagedAV.csv`: 平均化後時系列角速度の角速度を回転方向の切り替えを定量化．
  - `/scripts/functions/param.py`の`flag_evaluating_switching`を`True`にすると保存する
  - `ratation_analysis_main.py`のみで取得できる．
