# 0.5秒窓FFT振幅の包絡フィットと逆FFT

[ADR 0015](../adr/0015-poster-spectrum-envelope-fit.md)に従い、実測候補4細胞と合成対照2条件を同じ18パネル図に並べる。既存の長軸角・揺らぎ解析が保存したCSVとJSONを入力とし、元AVIやTIFFは再処理しない。実測由来の図・時系列・係数はGit管理外のローカル出力にのみ置く。

先に [`09_poster_fluctuation.py`](../../scripts/09_poster_fluctuation.py) の実測・合成解析を完了し、cleanなコミットから次を実行する。

```bash
python scripts/10_poster_one_over_f.py \
  --source-root outputs/poster_fluctuation \
  --output-root outputs/poster_fluctuation/2_simulation/one_over_f
```

入力は `20230607 No.06/08`、`20231005 No.01`、`20231107 No.06`、および200 fps・seed 81の `constant_angle_step_plus_noise` と `random_angle_noise` である。20231005 No.01の停止は既存FFTと同じ区間に含める。各ケースの0.5秒Gaussian窓の生SD/Mean、標準化SD/MeanのFFT振幅、逆FFT系列を1列に配置する。時系列の有効部分は元解析の最長連続区間を使い、フレーム対応・時刻間隔・保存済みFFTとの一致を検証する。

正のFFT振幅を対数周波数で9等分し、各帯域の振幅中央値を求める。最初の正周波数を `Δf=1/T_FFT` と固定し、`M(f)=A*((f+Δf)/(1 Hz))^(-α)` のAとαを帯域中央値の**対数振幅残差**が最小となるよう推定する。各非空帯域を等重みにし、αは0以上とする。FFTパネルは周波数を線形、振幅を対数目盛りとし、元のFFT・フィット帯域中央値・モデル曲線を重ねる。図中のR²は帯域中央値の対数振幅に対する値である。

逆FFTでは全正周波数ビンのモデル振幅に独立な一様乱数位相を付け、既定seed 84から列ごとに1ずつ加える。生成系列は平均0で、元の約5.5秒のFFT区間と同じ長さ・時刻に置く。この系列は**標準化SD/Meanのスペクトルに合わせた代理系列**であり、角度・角速度や、通常の「電力が1/f」のピンクノイズではない。位相乱数化で元の停止時刻や山谷は再現されない。

`outputs/poster_fluctuation/2_simulation/one_over_f/` に `six_case_one_over_f_0.5s.png`、各ケースの元FFT・モデル振幅・位相の `tables/*_fit.csv`、フィットに使う帯域の `tables/*_fit_bins.csv`、逆FFT時系列の `tables/*_inverse_fft.csv`、A・α・Δf・残差・seed・元フレーム範囲の `index.csv`、`README.txt`、`_run_records/` を保存する。モデルの傾きは窓処理と揺らぎ計算の影響も受け、図と逆FFT系列だけで生物学的な1/f過程や振動の有意性は判定しない。
