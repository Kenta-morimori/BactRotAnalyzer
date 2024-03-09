# Data directory
解析結果のディレクトリ

## 形式
`outputs`には，`data`ディレクトリに対応するディレクトリ名で解析結果が保存される．<br>

## angular_velocity
主に角速度解析の結果を保存している．<br>
- `angle_time-series.png`: 時系列角度．全重心座標に対し楕円フィッティングをし，楕円の中心と各重心座標の位置関係から角度を定義した．
- `angular-velocity_time-series.png`: 時系列角速度．各フレーム間の角度の変化にフレームレートを積算することで算出．反時計周りの角速度を正にした．
- `angular-velocity_time-series_averaged.png`: 平均化処理を行なった時系列角速度．特定の幅の窓関数をずらしながら平均値を算出することで平均化処理を行なった．

## centroid_coordinate
重心解析の結果を保存している．<br>
- `trajectory.png`: 時系列の重心座標．
- `x_coordinate.png`: 時系列重心x座標．
- `y_coordinate.png`: 時系列重心y座標．

## csvファイル
- `saved_centroid_coordinate.csv`: 時系列重心座標を保存．
- `switching_value.csv`: 回転方向の切り替えを定量化．時系列角速度の正・負の値を時計回り・反時計回りとしてカウントして，
- `sswitching_value_averagedAV.csv`: 平均化後時系列角速度の角速度を回転方向の切り替えを定量化．

