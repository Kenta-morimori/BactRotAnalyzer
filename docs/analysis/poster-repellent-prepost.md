# ポスター用忌避応答の上昇前後比較

[ADR 0025](../adr/0025-poster-repellent-prepost-comparison.md)に測定の判断と限界を記録する。一次選別結果を入力として、採用細胞の上昇前Low/High比と上昇後の診断的符号変化頻度をローカルで比較する。元画像・座標・具体的な細胞対応表はGitへ含めない。

## 入力

`scripts/11_poster_repellent_prepost.py` は `--manifest`、`--source-root`、`--output-dir` を受け取る。manifestはローカル専用JSONで、1試料につき次の項目を指定する。

```json
{
  "version": 1,
  "cases": [
    {
      "id": "synthetic_01",
      "label": "Synthetic 01",
      "input_kind": "repellent_bundle",
      "source": "relative/path/to/repellent_response",
      "sample_no": 1,
      "analysis_start_s": 0.0,
      "rise_time_s": 10.0,
      "analysis_end_s": 20.0,
      "invalid_source_frame_ranges": [[101, 103]]
    }
  ]
}
```

`input_kind` は `repellent_bundle` または `source_aligned_csv`。前者は現行フェーズ別出力の重心・中心・主角速度・背景輝度の4表を元フレームで結合し、時刻の一致を検証する。後者は `source_frame_1based,time_sec,raw_x_px,raw_y_px,center_x_px,center_y_px,angular_velocity_rad_s,background_intensity` を持つ保存系列を読む。`invalid_source_frame_ranges` は元フレームの両端を含み、元画像で確認した追跡失敗のみ指定する。空配列でもよい。

```bash
python3 scripts/11_poster_repellent_prepost.py \
  --manifest outputs/poster_fluctuation/4_repellent_response/private_candidates.json \
  --source-root outputs \
  --output-dir outputs/poster_fluctuation/4_repellent_response/pre_post_comparison
```

出力には細胞別の欠損・確認候補表、上昇前SD/Mean時系列とFFT、1秒ごとの符号変化表、窓幅別Low/High表、比較図、実行記録を含む。`summary.csv` は各細胞の元フレーム数、欠損数、0.5秒窓のLow/High比、符号変化回数、適格観測秒、1秒当たりの値をまとめる。確認候補フラグは自動除外ではない。
