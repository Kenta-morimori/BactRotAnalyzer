# ポスター用の長軸角・揺らぎ解析

## 回転区間だけの群比較

[ADR 0024](../adr/0024-poster-rotation-only-group-comparison.md)に従い、保存済みの実測時系列から最長の連続回転区間を選び、3条件の細胞を等重みで比較する。No.08（20230607）はユーザー指定で除外し、温度SJW46の重心周回角は群統計に入れない。以下はcleanなGitコミットから実行する。実測出力はGit管理外に置く。

```bash
python scripts/10_poster_rotation_only_stats.py \
  --measurements-root outputs/poster_fluctuation/1_rotational_measurements \
  --output-root outputs/poster_fluctuation/1_rotational_measurements/rotation_only_stats
```

`selection.csv` に元フレーム・時刻・最長区間・除外理由、`stop_intervals.csv` に停止候補、`cell_band_metrics.csv` に各窓のFFT分解能と低域/高域点数、`group_summary.csv` に細胞等重みの記述集計を保存する。`tables/` には選択区間の元フレーム付き時系列とFFTを置く。比較PNGは帯域比、細胞内最大値正規化、低域・高域振幅、60–80 Hz高域への感度を示す。AVIの対象と長軸追跡の目視判定は `visual_qc` 列で別管理し、未確認のまま確定採用を主張しない。

採否前の点検図 `original_full_record_<day>.png` は各細胞の元の6秒間の符号付き長軸角速度を左、元記録全体のLow/High比（0.1/0.5/1.0秒窓）を右に並べる。左図には群統計への採否を `USED` / `EXCLUDED` と明記し、採用細胞の揺らぎ評価に使った連続区間を緑で示す。`original_full_record_band_ratio.csv` に図の比を保存する。`selection_overview.png` は全24細胞について元記録・最長候補区間・停止候補・群統計への採否を示す。除外理由と候補区間の時刻は採否一覧図・CSVで確認する。これらの全記録比は回転区間だけの群比と区別する。

ポスター図の配置と角度FFTは [ADR 0022](../adr/0022-poster-figure-layout-and-angle-fft.md)、候補比較は [ADR 0023](../adr/0023-poster-candidate-comparison.md) に従う。個別PNGではx-y軌跡を物理尺度・枠とも正方形にし、角速度と揺らぎ・FFTを横長、帯域指標3種を各2:3の縦長で表示する。SD/Meanと揺らぎFFTの図中の線は0.1・0.5・1.0秒の3窓幅に絞るが、計算・CSV・帯域指標の6窓幅は保持する。折り返し角度のFFT生振幅を黒線で揺らぎFFTと同じ対数 `Amp` 縦軸に重ね、`tables/*_wrapped_angle_fft.csv` に保存する。右縦軸は設けない。直接速度ノイズ対照には角度がないため黒線は表示しない。折り返しの鋸歯状波と高調波を含む。角度FFTの振幅はrad、標準化SD/MeanのFFT振幅は無次元なので、同じ縦軸の高さは物理的な強度比較を意味しない。

SD/Meanと標準化SD/Meanの時系列図は、合成の0.5秒窓専用図を含めて横軸を0秒から記録の最大時刻までとする。

[基本測定](../adr/0009-poster-long-axis-fluctuation.md)、[表示・評価帯域の更新](../adr/0010-poster-phase-and-nyquist-band.md)、[固定帯域強度比](../adr/0011-poster-fixed-band-ratio.md)、[成分表示](../adr/0012-poster-band-components-and-steady-simulation.md)、[長軸角の合成対照](../adr/0013-poster-angle-noise-controls.md)、[実測速度による合成校正](../adr/0017-poster-empirical-speed-controls.md)、[独立速度対照の直接評価](../adr/0019-poster-direct-speed-control.md)、[30秒系列の追加](../adr/0020-poster-simulation-duration-extension.md)に従う。入力は現時点ではTIFFでなく二値AVI。対象は `data/20231005` のNo.1–7、`data/20230607` のNo.1–10、`data/20231107` のNo.1–7の計24細胞。両端の日付のNo.8–10は当時の解析用にNo.7を複製したAVIで、入力一覧に記録するが解析から除外する。各日付は同一条件の別細胞であり、実験反復と自動的にみなさない。

クリーンなコミットから、リポジトリのルートで実行する。

```bash
python scripts/09_poster_fluctuation.py --output-root outputs/poster_fluctuation
```

`--mode synthetic` で合成対照のみ、`--mode measured` で実測のみを実行する。合成のみの実行には採用3細胞の保存済み時系列CSVが必要。`--days` で日付を選択できる。合成対照は `--seed`、`--synthetic-rates`、`--synthetic-duration`（主時間長）、`--additional-synthetic-durations`、`--null-replicates` を指定できる。既定では6秒と30秒の両方を生成し、複数seed集計は主時間長6秒を用いる。独立速度ノイズの対照は既定で連続40 seed × 200/250 Hzの見かけのピーク分布も保存する。解析CLIは `analysis_run` によるclean Gitゲートと非公開の実行履歴を使う。完了した実行履歴は専用出力の `_run_records/` にもコピーする。

実測は `outputs/poster_fluctuation/1_rotational_measurements/<day>/No.XX*.png` に各細胞1枚ずつ保存する。代表3細胞のPNGには条件名と `_representative` を付け、`index.csv` の `plot_file` に実際のファイル名を記録する。日付ディレクトリは `20230607`、`20231005`、`20231107` で、各日付の `tables/` に対応する時系列・FFT・品質メタデータを保存する。各実測PNGの12パネルには t-x、t-y、x-y、推定回転位相、符号付き長軸角速度、絶対値長軸角速度、図上では3窓幅を重ねたSD/Meanと標準化系列、FFT、固定帯域のLow/High・Lowのみ・Low−Highを横並びで載せる。角度は連続区間ごとに初期角を0とした相対角度を `[-π, π)` に折り返して描き、境界の線は切る。頭尾を識別した物理的な向きではない。角速度と時系列CSVには折り返し前の相対角度も残す。図はArialを使い、軸ラベル・目盛・タイトルを拡大し、凡例は各グラフの外に置く。角度FFTの黒線は透明度を上げて重なりを見やすくする。実測の周回角速度・揺らぎ・FFTは出力しない。

FFTの横軸上限は各AVIのFrame Rateの半分（200 fpsなら100 Hz、250 fpsなら125 Hz）。既存FFTが返す正の周波数点は上限未満で、振幅は対数目盛りの図に描く。振幅0は図に描かずCSVに保持する。長軸のFFT値とピーク判定の計算は維持する。乱数のみの対照のピーク/中央値比は旧0–20 Hzではなく全正周波数帯で再計算し、列名は `peak_to_median_full_band` とする。`1_rotational_measurements/index.csv` は実測の品質一覧で、`plot_file` は日付ディレクトリからの相対パスを示す。`selection_template.csv` は完全一致する重複AVIを示し、異なる動画の採否は目視確認待ちとする。手動採否は同ディレクトリの `selection.csv` に記入し、CLIは上書きしない。

強度比は各窓幅の標準化SD/MeanのFFTから、`0 < f < 1 Hz` の平均振幅を `80 <= f < 100 Hz` の平均振幅で割る。DCは除外し、帯域に点がないか高域平均が0なら比を欠測にする。全24細胞の比、Low−High、両帯域の平均振幅・FFT点数・帯域境界は日付別 `tables/No.XX_band_amplitude_ratio.csv` に保存する。Low/Highの縦軸は対数目盛り、LowとLow−Highは線形目盛りで表示する。20230607 No.06、20231005 No.01、20231107 No.06と温度解析のSJW46 23°C No.06を `1_rotational_measurements/candidate_band_amplitude_ratio.png` で重ねる。各細胞・各指標の6窓幅系列をその系列の有限な最大値で割り、縦軸最大値を1として窓幅による形を比較する。比較図は3パネルで、横軸は `Window width [s]`、縦軸は `Amp ratio` とする。凡例は図の右外側にSJW46、SJW46/MotAB、YSC2123/MotAB、YcgR-tetの順で縦に置く。CSV内では温度を識別できるよう先頭の条件名「SJW46: 23°C」を維持する。温度23°Cの系列は保存済み中心を使う重心周回角、他の3系列は菌体長軸角から得た揺らぎである。角度定義が異なることを図中と `tables/candidate_band_amplitude_ratio.csv` に明記する。CSVは元の振幅・比と正規化値の両方を保持する。温度23°Cの帯域表がまだない場合は3細胞で図を作り、後から温度解析を実行して本CLIを再実行すると4系列になる。20230607 No.08の個別解析は残し、候補比較から除く。20231005 No.01は停止を含む全時系列を使用するため、停止の影響を受け得る。候補比較は最終採否や有意差検定ではない。`selection.csv` はCLIから変更しない。

合成の目標は上記3候補の細胞別平均絶対角速度の等重み平均 `μ` と、細胞内速度SDの等重み平均 `σ` とする。`constant_angle_step_plus_noise` は `θ_i=μt_i+ε_i`、`ε_i` は独立な正規角度誤差でSDを `σ/(√2 fps)` とする。この条件は180°周期の観測長軸角から実測と同じ差分で角速度を再計算する。`random_speed_noise` は各区間の速度を独立な `μ+N(0,σ²)` として直接解析し、角度は生成・積分・保存しない。速度は区間末尾のフレームに対応し、初回フレームは欠測とする。既定は200/250 fps、各6秒・30秒、seed 81。実際の平均・SDは有限標本と絶対値化で目標から少しずれる。

合成代表例の `simulation_*.png` と独立速度ノイズの複数seed比較 `simulation_random_speed_noise_ensemble.png` は `outputs/poster_fluctuation/2_simulation/` に保存する。6秒系列は従来のファイル名を維持し、30秒系列には `_30s_` を付ける。合成索引には両時間長の8行を記録する。5条件の1/f比較は引き続き6秒系列を使う。合成代表例は各ケース12枠で、角度誤差条件の位相、符号付き/絶対値角速度、図上では3窓幅のSD/Mean・標準化SD/Mean・FFT、その下に0.5秒窓だけの同3図、3つの固定帯域指標を表示する。直接速度条件の角度枠には何も描かない。重心座標・周回角速度は生成・図示・保存しない。対応する6窓幅の値は `tables/simulation_*_band_amplitude_ratio.csv` に保存し、`index.csv` の `band_ratio_file` から参照する。校正元の3細胞の統計は `tables/simulation_calibration_cells.csv`、時系列・長軸FFT・品質メタデータと複数seed集計も同ディレクトリの `tables/` に保存する。実行履歴は出力ルートの `_run_records/` に保存する。旧配置の `selection.csv` は新配置へ移し、既存の手動判定を保持する。

揺らぎは1フレームずつ中心を動かすGaussian窓で `SD(abs(長軸角速度))/Mean(abs(長軸角速度))` を求める。窓幅は0.1/0.2/0.5/1.0/1.5/2.0秒。窓全体が記録内にあり、3点以上の有効値があり、欠測や時刻gapを含まない位置だけを計算する。各窓幅の有効な揺らぎ系列全体をz-score標準化し、最長の連続有効区間をFFTにかける。図には0.1・0.5・1.0秒の3窓幅だけ重ねる。採用前に元AVIで対象と楕円長軸を確認し、主角速度の欠測率と折返し、窓幅を変えたピークの変化、合成対照で生じる見かけのピークを点検する。採否表はローカルに保持し、実測座標・図・再構成可能な系列をIssueやPRに掲載しない。
