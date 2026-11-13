# ポスター用の長軸角・揺らぎ解析

[測定の判断と限界](../adr/0009-poster-long-axis-fluctuation.md)に従う。対象は `data/20231005`、`data/20230607`、`data/20231107` の各10 AVI。各日付は同一条件の別細胞であり、実験反復と自動的にみなさない。

クリーンなコミットから、リポジトリのルートで実行する。

```bash
python scripts/09_poster_fluctuation.py --output-root outputs/poster_fluctuation
```

`--mode synthetic` で合成対照のみ、`--mode measured` で実測のみを実行する。`--days` で日付を選択できる。合成対照は `--seed`、`--synthetic-rates`、`--synthetic-duration` を指定できる。解析CLIは `analysis_run` によるclean Gitゲートと非公開の実行履歴を使う。

`outputs/poster_fluctuation/` の `measured/<day>/No.XX/` に各細胞の `overview.png`、`time_series.csv`、長軸と周回角のFFT、測定区間・欠測数・設定の `metadata.json` を保存する。`synthetic/<model>_<rate>hz_seed<seed>/` は同じ構成に真の角速度と真の座標を加える。`measured/<day>/condition_summary.png` は10細胞の0.2秒窓z-scoreとFFTを同じ縦軸・0–20 Hz横軸で並べる。`index.csv` は品質の一覧で、自動採否ではない。

一覧図の揺らぎ・FFTは代表の0.2秒窓。全周波数の振幅はCSVに残す。CSVには全6窓幅の系列とFFTを保存する。採用前に元動画で対象と楕円長軸を確認し、主角速度の欠測率と角速度の折返し、周回角との関係、窓幅を変えたピークの変化、合成対照で生じる見かけのピークを点検する。採否表はローカルに保持し、実測座標・図・再構成可能な系列をIssueやPRに掲載しない。
