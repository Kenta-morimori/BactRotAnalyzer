# ポスター用の長軸角・揺らぎ解析

[測定の判断と限界](../adr/0009-poster-long-axis-fluctuation.md)に従う。対象は `data/20231005` のNo.1–7、`data/20230607` のNo.1–10、`data/20231107` のNo.1–7の計24細胞。両端の日付のNo.8–10は当時の解析用にNo.7を複製したAVIで、入力一覧に記録するが解析から除外する。各日付は同一条件の別細胞であり、実験反復と自動的にみなさない。

クリーンなコミットから、リポジトリのルートで実行する。

```bash
python scripts/09_poster_fluctuation.py --output-root outputs/poster_fluctuation
```

`--mode synthetic` で合成対照のみ、`--mode measured` で実測のみを実行する。`--days` で日付を選択できる。合成対照は `--seed`、`--synthetic-rates`、`--synthetic-duration`、`--null-replicates` を指定できる。乱数のみの対照は既定で連続40 seed × 200/250 Hzの見かけのピーク分布も保存する。解析CLIは `analysis_run` によるclean Gitゲートと非公開の実行履歴を使う。完了した実行履歴は専用出力の `_run_records/` にもコピーする。

実測は `outputs/poster_fluctuation/1_rotational_measurements/<day>/No.XX.png` に各細胞1枚ずつ保存する。日付ディレクトリは `20230607`、`20231005`、`20231107` で、各日付の `tables/` に対応する時系列・FFT・品質メタデータを保存する。各PNGには t-x、t-y、x-y、長軸の連続区間ごとに初期角を0とした相対角度、符号付きと絶対値の長軸角速度、6窓幅のSD/Meanと標準化系列、各窓幅のFFTを同一図に収める。FFTの横軸は線形、振幅の縦軸は対数目盛りとし、振幅0の点のみ図から除く。CSVの振幅値やピーク判定は変えない。`1_rotational_measurements/index.csv` は実測の品質一覧で、`plot_file` は日付ディレクトリからの相対パスを示す。`selection_template.csv` は完全一致する重複AVIを示し、異なる動画の採否は目視確認待ちとする。手動採否は同ディレクトリの `selection.csv` に記入し、CLIは上書きしない。

合成代表例の `simulation_*.png` と乱数のみの複数seed比較 `simulation_random_only_ensemble.png` は `outputs/poster_fluctuation/2_simulation/` に保存する。対応する時系列・FFT・品質メタデータと複数seed集計は同ディレクトリの `tables/`、合成代表例の一覧は `index.csv` に保存する。実行履歴は出力ルートの `_run_records/` に保存する。旧配置の `selection.csv` は新配置へ移し、既存の手動判定を保持する。

図の揺らぎ・標準化系列・FFTには全6窓幅を重ねる。FFT図の横軸は0–20 Hzに限定し、全周波数の振幅はCSVに残す。採用前に元動画で対象と楕円長軸を確認し、主角速度の欠測率と角速度の折返し、周回角との関係、窓幅を変えたピークの変化、合成対照で生じる見かけのピークを点検する。採否表はローカルに保持し、実測座標・図・再構成可能な系列をIssueやPRに掲載しない。
