# 0.5秒窓FFT振幅の1/f形フィット

[ADR 0014](../adr/0014-poster-one-over-f-amplitude-surrogates.md)に従い、実測候補4細胞と合成対照2条件を同じ18パネル図に並べる。既存の長軸角・揺らぎ解析が保存したCSVとJSONを入力とし、元AVIやTIFFは再処理しない。実測由来の図・時系列・係数はGit管理外のローカル出力にのみ置く。

先に [`09_poster_fluctuation.py`](../../scripts/09_poster_fluctuation.py) の実測・合成解析を完了し、cleanなコミットから次を実行する。

```bash
python scripts/10_poster_one_over_f.py \
  --source-root outputs/poster_fluctuation \
  --output-root outputs/poster_fluctuation/3_one_over_f
```

入力は `20230607 No.06/08`、`20231005 No.01`、`20231107 No.06`、および200 fps・seed 81の `constant_angle_step_plus_noise` と `random_angle_noise` である。20231005 No.01の停止は既存FFTと同じ区間に含める。各ケースの0.5秒Gaussian窓の生SD/Mean、標準化SD/MeanのFFT振幅、逆FFT系列を1列に配置する。時系列の有効部分は元解析の最長連続区間を使い、フレーム対応・時刻間隔・保存済みFFTとの一致を検証する。

正の周波数ビンすべてを、線形のFFT振幅で `A/(f+1 Hz)` に切片なしの最小二乗で当てはめる。`1 Hz` は固定オフセットで、推定するのは `A` だけである。逆FFTでは各ビンに独立な一様乱数位相を付け、既定seed 84から列ごとに1ずつ加える。生成系列は平均0で、元の約5.5秒のFFT区間と同じ長さ・時刻に置く。これは**FFT振幅の1/f形**であり、通常の「電力が1/f」のピンクノイズとは異なる。位相乱数化で元の停止時刻や山谷は再現されない。

`outputs/poster_fluctuation/3_one_over_f/` に `six_case_one_over_f_0.5s.png`、各ケースの周波数・実測/フィット振幅・位相の `tables/*_fit.csv`、逆FFT時系列の `tables/*_inverse_fft.csv`、係数・RMSE・R²・seed・元フレーム範囲の `index.csv`、`README.txt`、`_run_records/` を保存する。R²の低いケースは1/f形の当てはまりが弱いことを示す。図と逆FFT系列だけで振動の有意性は判定しない。
