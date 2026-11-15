# ポスター用忌避応答候補の一次選別

既存の忌避応答出力から、試料別の回転・背景輝度図と候補一覧を私有ディレクトリに作る。測定・停止の定義と限界は [ADR 0005](../adr/0005-poster-repellent-screening.md) に記載する。

```bash
python3 scripts/11_poster_repellent_screening.py \
  --source-root /path/to/outputs/repellent-response \
  --raw-root /path/to/data/repellent-response \
  --output-root /path/to/outputs/poster_fluctuation/4_repellent_response
```

一部の出力だけ試す場合は `--datasets 23 20260618` のように指定する。実行にはcommit済みのcleanなGit作業ツリーが必要で、Git・入力hash・環境・結果は`outputs/_runs/`と出力先の`_run_records/`に保存する。

出力先には`source_inventory.csv`、`index.csv`、`review_first.csv`、`selection_template.csv`、試料別の`<dataset>/No.XX.png`を置く。`review_first.csv`は図を優先確認するための暫定抽出で、採用結果ではない。`selection.csv`を手動で作成していた場合は上書きしない。再出力・未処理の原データは`source_inventory.csv`で区別する。元の試料番号は再採番しない。輝度上昇前SD/Mean・FFT、上昇後のスイッチング、停止の確定判定は今回行わない。実測図と系列はGitへ追加しない。
