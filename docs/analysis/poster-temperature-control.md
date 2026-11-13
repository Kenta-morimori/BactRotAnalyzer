# SJW46温度条件別のポスター解析

[温度解析の決定](../adr/0014-poster-temperature-orbit.md)に従う。10/23/30°Cの既存解析出力 `outputs/SJW46_temp=10/23/30` から、各細胞の元フレーム番号・時刻、中心相対重心、補正後・補正前回転中心を読み込む。元動画の再追跡は行わない。

クリーンなコミットからリポジトリルートで実行する。

```bash
python scripts/10_poster_temperature_control.py
```

入力を別のローカルチェックアウトに置く場合は `--source-root`（旧 `outputs/`）、`--config-root`（旧 `data/`）、出力先を変える場合は `--output-root` を指定する。解析CLIは `analysis_run` で入力ファイルの同一性、Git状態、パラメータ、実行結果を非公開ログへ記録する。

出力先は既定で `outputs/poster_fluctuation/3_Temperature_Control/`。`10C/23C/30C/No.XX.png` に各細胞の重心と回転中心のt-x、t-y、x-y、半径、周回角、符号付き・絶対値角速度、6窓幅の揺らぎと標準化系列、FFT、Low/High比を載せる。対応する `tables/` に元フレーム番号付き時系列、FFT、窓幅ごとの帯域値、メタデータを保存する。ルートの `cell_summary.csv` は細胞単位の平均速さ、平均半径、0.5秒窓比、gap・中心補正の件数と出力相対パスをまとめる。`window_band_amplitude_ratio.csv` は全細胞・全窓幅の低域、高域、差、比、FFT点数を保持する。

比較図は `compare_mean_angular_speed.png`、`compare_mean_rotation_radius.png`、`compare_low_frequency_relative_strength.png`、`compare_speed_vs_low_frequency_strength.png` の4枚。条件別の個々の細胞と細胞平均を区別する。速さ対強度は0.5秒窓比を使う。細胞間の記述的比較であり、実験反復としての有意差検定は行わない。

`centroid_coordinate.csv` の座標はすでに補正後中心を引いた値（µm）である。図の絶対重心はそれに補正後中心を足して表示する。半径と角度は中心相対座標から算出する。画像y軸は下向きなので角度は `atan2(-y,x)` とし、反時計回りを正とする。実時刻のgapをまたぐ角速度と揺らぎ窓は欠測にする。平均速さは有効な絶対値角速度、平均半径は有効な距離のフレーム平均。固定帯域は `0<f<1 Hz` と `80<=f<100 Hz` の平均FFT振幅で、窓幅ごとの比を出す。元の中心補正の影響や追跡品質は別途確認が必要。実測由来の図・時系列・実行ログはGit管理外に保つ。
