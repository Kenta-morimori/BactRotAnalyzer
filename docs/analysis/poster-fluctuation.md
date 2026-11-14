# ポスター用の長軸角・揺らぎ解析

[基本測定](../adr/0009-poster-long-axis-fluctuation.md)、[表示・評価帯域の更新](../adr/0010-poster-phase-and-nyquist-band.md)、[固定帯域強度比](../adr/0011-poster-fixed-band-ratio.md)、[成分表示](../adr/0012-poster-band-components-and-steady-simulation.md)、[長軸角の合成対照](../adr/0013-poster-angle-noise-controls.md)、[実測速度による合成校正](../adr/0017-poster-empirical-speed-controls.md)、[独立速度対照の直接評価](../adr/0019-poster-direct-speed-control.md)に従う。入力は現時点ではTIFFでなく二値AVI。対象は `data/20231005` のNo.1–7、`data/20230607` のNo.1–10、`data/20231107` のNo.1–7の計24細胞。両端の日付のNo.8–10は当時の解析用にNo.7を複製したAVIで、入力一覧に記録するが解析から除外する。各日付は同一条件の別細胞であり、実験反復と自動的にみなさない。

クリーンなコミットから、リポジトリのルートで実行する。

```bash
python scripts/09_poster_fluctuation.py --output-root outputs/poster_fluctuation
```

`--mode synthetic` で合成対照のみ、`--mode measured` で実測のみを実行する。合成のみの実行には採用3細胞の保存済み時系列CSVが必要。`--days` で日付を選択できる。合成対照は `--seed`、`--synthetic-rates`、`--synthetic-duration`、`--null-replicates` を指定できる。独立速度ノイズの対照は既定で連続40 seed × 200/250 Hzの見かけのピーク分布も保存する。解析CLIは `analysis_run` によるclean Gitゲートと非公開の実行履歴を使う。完了した実行履歴は専用出力の `_run_records/` にもコピーする。

実測は `outputs/poster_fluctuation/1_rotational_measurements/<day>/No.XX.png` に各細胞1枚ずつ保存する。日付ディレクトリは `20230607`、`20231005`、`20231107` で、各日付の `tables/` に対応する時系列・FFT・品質メタデータを保存する。各実測PNGの12パネルには t-x、t-y、x-y、推定回転位相、符号付き長軸角速度、絶対値長軸角速度、6窓幅を重ねたSD/Meanと標準化系列、FFT、固定帯域のLow/High・Lowのみ・Low−Highを横並びで載せる。角度は連続区間ごとに初期角を0とした相対角度を `[-π, π)` に折り返して描き、境界の線は切る。頭尾を識別した物理的な向きではない。角速度と時系列CSVには折り返し前の相対角度も残す。軸ラベル・目盛・凡例は拡大した。実測の周回角速度・揺らぎ・FFTは出力しない。

FFTの横軸上限は各AVIのFrame Rateの半分（200 fpsなら100 Hz、250 fpsなら125 Hz）。既存FFTが返す正の周波数点は上限未満で、振幅は対数目盛りの図に描く。振幅0は図に描かずCSVに保持する。長軸のFFT値とピーク判定の計算は維持する。乱数のみの対照のピーク/中央値比は旧0–20 Hzではなく全正周波数帯で再計算し、列名は `peak_to_median_full_band` とする。`1_rotational_measurements/index.csv` は実測の品質一覧で、`plot_file` は日付ディレクトリからの相対パスを示す。`selection_template.csv` は完全一致する重複AVIを示し、異なる動画の採否は目視確認待ちとする。手動採否は同ディレクトリの `selection.csv` に記入し、CLIは上書きしない。

強度比は各窓幅の標準化SD/MeanのFFTから、`0 < f < 1 Hz` の平均振幅を `80 <= f < 100 Hz` の平均振幅で割る。DCは除外し、帯域に点がないか高域平均が0なら比を欠測にする。全24細胞の比、Low−High、両帯域の平均振幅・FFT点数・帯域境界は日付別 `tables/No.XX_band_amplitude_ratio.csv` に保存する。Low/Highの縦軸は対数目盛り、LowとLow−Highは線形目盛りで表示する。20230607 No.06、20231005 No.01、20231107 No.06の3候補は `1_rotational_measurements/candidate_band_amplitude_ratio.png` で重ね、数値を同階層の `tables/candidate_band_amplitude_ratio.csv` に保存する。比較図は3パネルで、3候補をすべて含む実測実行時に生成する。20230607 No.08の個別解析は残し、候補比較から除く。20231005 No.01は停止を含む全時系列を使用するため、停止の影響を受け得る。候補比較は最終採否や有意差検定ではない。`selection.csv` はCLIから変更しない。

合成の目標は上記3候補の細胞別平均絶対角速度の等重み平均 `μ` と、細胞内速度SDの等重み平均 `σ` とする。`constant_angle_step_plus_noise` は `θ_i=μt_i+ε_i`、`ε_i` は独立な正規角度誤差でSDを `σ/(√2 fps)` とする。この条件は180°周期の観測長軸角から実測と同じ差分で角速度を再計算する。`random_speed_noise` は各区間の速度を独立な `μ+N(0,σ²)` として直接解析し、角度は生成・積分・保存しない。速度は区間末尾のフレームに対応し、初回フレームは欠測とする。既定は200/250 fps、各6秒、seed 81。実際の平均・SDは有限標本と絶対値化で目標から少しずれる。

合成代表例の `simulation_*.png` と独立速度ノイズの複数seed比較 `simulation_random_speed_noise_ensemble.png` は `outputs/poster_fluctuation/2_simulation/` に保存する。合成代表例は各ケース9枠で、角度誤差条件の位相、符号付き/絶対値角速度、6窓幅のSD/Mean・z-score・FFT、3つの固定帯域指標を表示する。直接速度条件の角度枠には何も描かない。重心座標・周回角速度は生成・図示・保存しない。対応する6窓幅の値は `tables/simulation_*_band_amplitude_ratio.csv` に保存し、`index.csv` の `band_ratio_file` から参照する。校正元の3細胞の統計は `tables/simulation_calibration_cells.csv`、時系列・長軸FFT・品質メタデータと複数seed集計も同ディレクトリの `tables/` に保存する。実行履歴は出力ルートの `_run_records/` に保存する。旧配置の `selection.csv` は新配置へ移し、既存の手動判定を保持する。

揺らぎは1フレームずつ中心を動かすGaussian窓で `SD(abs(長軸角速度))/Mean(abs(長軸角速度))` を求める。窓幅は0.1/0.2/0.5/1.0/1.5/2.0秒。窓全体が記録内にあり、3点以上の有効値があり、欠測や時刻gapを含まない位置だけを計算する。各窓幅の有効な揺らぎ系列全体をz-score標準化し、最長の連続有効区間をFFTにかける。図には全6窓幅を重ねる。採用前に元AVIで対象と楕円長軸を確認し、主角速度の欠測率と折返し、窓幅を変えたピークの変化、合成対照で生じる見かけのピークを点検する。採否表はローカルに保持し、実測座標・図・再構成可能な系列をIssueやPRに掲載しない。
