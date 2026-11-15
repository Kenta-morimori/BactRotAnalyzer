# SJW46温度条件別のポスター解析

ポスター用の個別図は [ADR 0022](../adr/0022-poster-figure-layout-and-angle-fft.md) に従い、x-y軌跡を正方形、角速度・揺らぎ・FFTを横長、Low/High・Lowのみ・Low−Highの3パネルを各2:3の縦長にする。揺らぎ・FFTの図中の線は0.1・0.5・1.0秒窓だけとし、6窓幅の計算とCSVは保持する。折り返し周回角のFFTは右側のrad単位の対数縦軸に黒線で重ね、`tables/No.XX_wrapped_angle_fft.csv` に保存する。折り返しによる高調波を含み、揺らぎFFTの振幅とは単位が異なる。温度別の単一指標図は横幅を縮め、点と文字を拡大する。

[温度解析の決定](../adr/0014-poster-temperature-orbit.md)に従う。10/23/30°Cの既存解析出力 `outputs/SJW46_temp=10/23/30` から、各細胞の元フレーム番号・時刻、中心相対重心、補正後・補正前回転中心を読み込む。元動画の再追跡は行わない。

クリーンなコミットからリポジトリルートで実行する。

```bash
python scripts/10_poster_temperature_control.py
```

入力を別のローカルチェックアウトに置く場合は `--source-root`（旧 `outputs/`）、`--config-root`（旧 `data/`）、出力先を変える場合は `--output-root` を指定する。解析CLIは `analysis_run` で入力ファイルの同一性、Git状態、パラメータ、実行結果を非公開ログへ記録する。

保存済み中心で算出した細胞ごとの平均回転半径が0.2 µm以下の細胞だけを比較する場合は、同じCLIに `--max-mean-radius-um 0.2 --radius-axis-max-um 0.5` を付ける。既定の全細胞出力は維持し、選択結果を `3_Temperature_Control/mean_radius_le_0.2um/` に分ける。このディレクトリには採用細胞の個別図・表と5比較図を出し、全入力細胞の採否と理由を `selection.csv` に記録する。閾値ちょうどの細胞は含め、平均半径が欠測の細胞は除く。これは任意の感度解析であり、撮影品質や生物学的採否の判定を意味しない。

同じ半径制限版で `--other-window-speed-plots` を追加すると、0.5秒以外の0.1・0.2・1.0・1.5・2.0秒窓について、細胞平均絶対角速度と各窓のLow/High比の散布図を5枚、5幅をまとめた一覧図を1枚追加する。窓幅ごとに線形の縦軸範囲を設定する。既存の `compare_speed_vs_low_frequency_strength.png` は0.5秒窓の図として残る。

出力先は既定で `outputs/poster_fluctuation/3_Temperature_Control/`。`10C/23C/30C/No.XX.png` に各細胞の重心と回転中心のt-x、t-y、x-y、半径、周回角、符号付き・絶対値角速度、3窓幅の揺らぎと標準化系列・FFT、折り返し角度FFT、6窓幅のLow/High・Lowのみ・Low−Highを載せる。対応する `tables/` に元フレーム番号付き時系列、FFT、窓幅ごとの帯域値、メタデータを保存する。ルートの `cell_summary.csv` は細胞単位の平均速さ、平均半径、0.5秒窓比、gap・中心補正の件数と出力相対パスをまとめる。`window_band_amplitude_ratio.csv` は全細胞・全窓幅の低域、高域、差、比、FFT点数を保持する。

比較図は `compare_mean_angular_speed.png`、`compare_mean_rotation_radius.png`、6窓幅の `compare_low_frequency_relative_strength.png`、0.5秒窓だけを温度軸で比較する `compare_low_frequency_relative_strength_0.5s_by_temperature.png`、`compare_speed_vs_low_frequency_strength.png` の5枚。条件別の個々の細胞と細胞平均を区別する。平均半径の縦軸は既定で0–1.0 µmとし、0.2 µm以下の感度解析では上記オプションで0–0.5 µmにする。Low/High比の縦軸は比較図で線形表示にする。速さ対強度は0.5秒窓比を使う。細胞間の記述的比較であり、実験反復としての有意差検定は行わない。

`centroid_coordinate.csv` の座標はすでに補正後中心を引いた値（µm）である。図の絶対重心はそれに補正後中心を足して表示する。半径と角度は中心相対座標から算出する。画像y軸は下向きなので角度は `atan2(-y,x)` とし、反時計回りを正とする。実時刻のgapをまたぐ角速度と揺らぎ窓は欠測にし、図の線もgapで切る。平均速さは有効な絶対値角速度、平均半径は有効な距離のフレーム平均。固定帯域は `0<f<1 Hz` と `80<=f<100 Hz` の平均FFT振幅で、窓幅ごとの比を出す。元の中心補正の影響や追跡品質は別途確認が必要。実測由来の図・時系列・実行ログはGit管理外に保つ。
