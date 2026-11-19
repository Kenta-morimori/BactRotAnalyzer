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
  --manifest outputs/poster_fluctuation/4_repellent_response/private_prepost_candidates.json \
  --source-root outputs \
  --output-dir outputs/poster_fluctuation/4_repellent_response \
  --representative-case-id 23_No14
```

出力は `4_repellent_response/` 直下に図・集計表・`README_prepost.txt`、その `tables/` に細胞別CSVを置く。既存の `README.txt` や他の解析ディレクトリは保持する。細胞別の欠損・確認候補表、上昇前SD/Mean時系列とFFT、1秒ごとの非重複符号変化表、1フレームずつ動かす1秒窓の符号変化時系列、窓幅別Low/High表、実行記録を含む。`summary.csv` は各細胞の元フレーム数、欠損数、0.5秒窓のLow/High比、符号変化回数、適格観測秒、1秒当たりの値をまとめる。確認候補フラグは自動除外ではない。

`centroid_mask_qc.png` は元の重心、残した重心、回転中心を重ね、除外した元の重心を赤い×と薄赤の時間帯で示す。単位は入力に合わせてµmまたはpx。`candidate_prepost_overview.png` は試料ごとに角速度、背景輝度、1秒窓の符号変化率を上下へ連結して時間軸をそろえ、CCW正を青、CW負を橙、除外区間を薄赤で示す。スライド窓は元フレーム時刻を中心として1フレームずつ進め、欠測や記録gapをまたぐ隣接対は数えない。解析区間端の不完全な窓は非表示とし、各窓の分母は `tables/*_sliding_1s_sign_changes.csv` の `eligible_seconds` に残す。

`--representative-case-id` でmanifest内の代表試料を指定すると、`representative_centroid.png` と `representative_response.png` を単独で出力する。試料名と凡例を省き、軸名・単位・目盛りは残す。複数試料の定量比較は `summary.csv` と `pre_low_high_by_window.csv` に保存する。符号変化は診断値であり、生物学的な回転方向切替を確定する値ではない。
