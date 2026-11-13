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

逆FFTでは全正周波数ビンのモデル振幅に独立な一様乱数位相を付け、既定seed 84から列ごとに1ずつ加える。生成系列は平均0で、元の約5.5秒のFFT区間と同じ長さ・時刻に置く。この系列は**標準化SD/MeanのFFT振幅に合わせた代理系列**であり、角度・角速度ではない。色名によるノイズ分類は行わない。位相乱数化で元の停止時刻や山谷は再現されない。

`outputs/poster_fluctuation/2_simulation/one_over_f/` に `six_case_one_over_f_0.5s.png`、各ケースの元FFT・モデル振幅・位相の `tables/*_fit.csv`、フィットに使う帯域の `tables/*_fit_bins.csv`、逆FFT時系列の `tables/*_inverse_fft.csv`、A・α・Δf・残差・seed・元フレーム範囲の `index.csv`、`README.txt`、`_run_records/` を保存する。モデルの傾きは窓処理と揺らぎ計算の影響も受け、図と逆FFT系列だけで生物学的な1/f過程や振動の有意性は判定しない。

## フィット感度の比較

採用式を変更する前に、1条件の同一FFTへ、対数周波数帯域の要求数を9・18・36とし、αを1に固定する場合と推定する場合を2行×3列に並べられる。例は20230607 No.08。黄色の点は実際にフィットへ使った各非空帯域の振幅中央値であり、帯域ごとに元FFT点数は異なる。対数振幅残差を各帯域で等重みとし、`Δf` は全パネルで最初の正周波数に固定する。`log RMSE` は各パネル自身の帯域中央値に対する値で、帯域数をまたぐ優劣の判定には使わない。この比較は主解析の6条件図・逆FFT系列を変更しない。

```bash
python scripts/10_poster_one_over_f.py \
  --source-root outputs/poster_fluctuation \
  --output-root outputs/poster_fluctuation/2_simulation/one_over_f \
  --fit-sensitivity-case 20230607_No08
```

`fit_sensitivity_20230607_No08_0.5s.png` と実行履歴をGit管理外に保存する。図は実測に由来するため、リポジトリやPRへ添付しない。

高周波側も線形の周波数軸で等密度に参照点を置く比較は、同じ0.5秒窓FFTを0から2・1・0.5 Hz幅に分け、非空帯域の振幅中央値へαを自由にした同じ式を当てはめる。上段は全周波数、下段は0–10 Hzの拡大である。各帯域中央値は等重みなので、帯域を狭めるほど点数の多い高周波側がフィットに強く効く。この比較も主解析の採用式・逆FFT系列を変更しない。

```bash
python scripts/10_poster_one_over_f.py \
  --source-root outputs/poster_fluctuation \
  --output-root outputs/poster_fluctuation/2_simulation/one_over_f \
  --equal-hz-sensitivity-case 20230607_No08
```

`fit_equal_hz_20230607_No08_0.5s.png` と実行履歴をGit管理外に保存する。
