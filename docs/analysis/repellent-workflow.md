# 忌避応答解析の実行と出力

重要な設計判断は [ADR一覧](../adr/README.md) に記録する。Wordは実測の説明資料、本文書は現在の実行方法・保存契約である。

生TIFFからの重心抽出、config.iniなしの入力設定、TIFF/AVIを切り替えた再解析は[画像入力の接続手順](repellent-image-input.md)を参照する。明示入力ではユーザー作成のMean CSVを読み、[持続上昇検出](background-rise.md)で背景上昇点を検出する。元フレームによる手動境界も指定できる。

## フェーズと既定手法

| フェーズ | 処理・保存する結果 |
|---|---|
| 入力・重心 | 既定はAVIの二値領域内の面積重心。元フレーム・時刻・欠測を保存。重心平滑化なし。旧輪郭平均を比較入力に使うときは抽出法を明記した入力manifestを指定する |
| 背景 | 提供Meanの頑健な持続上昇検出・手動境界設定。元フレームを境界として前後分割する |
| 中心推定 | 局所楕円・時間変化モデル。支持窓の元観測を全点利用し、20フレームごとの評価と短区間補間。直接推定・棄却・短補間を区別する |
| 中心補完 | 直前・直後の有限ブロックの直接推定点を各最大3点平均し、平均時刻で線形接続。片側は平均保持。支持なしは欠測。既存中心・元欠測・記録境界を変更しない |
| 角速度 | 隣接2フレームの中心相対角度差÷実時間差。CCW正・CW負、平滑化なし。欠測・gap・対象境界を越えない。補完中心による測定は暫定参照として区別する |
| 揺らぎ・FFT | 背景上昇前。元観測間隔のスライドSD、既存のGaussian重み・SD/平均・標準化・周波数特徴定義を保持。FFT表示は両対数 |
| 切替診断 | 背景上昇後。隣接符号変化候補と観測量を保存。物理的反転の確定・安定化時間は未採用なら未判定 |

中心・半径の同時変動、位相同期の半径変動等は実測物理中心の精度保証の対象外。長時間停止中の補完も前後の中心を繋ぐ仮定である。推定・補完の設定と支持点は必ず結果と一緒に読む。

## 通常の実行

リポジトリのルートで実行する。

解析CLIはcommit済みのcleanなGit作業ツリーを要求する。ブランチ・SHA・入力hash・環境・終了状態は
`outputs/_runs/` に保存する。[実行履歴の契約](../development-environment.md)と[ADR 0004](../adr/0004-execution-provenance.md)を参照。

```bash
python3 scripts/04_repellent_response_analysis.py --day repellent-response/<dataset>
python3 scripts/04_repellent_response_analysis.py --day repellent-response/<dataset> --all-frames --output-suffix comparison
```

`utils/param.py` の `center_estimator="local_conic"` が既定。推定窓・診断条件・評価間隔は `repellent_center_parameters`、補完点数は `repellent_center_completion_points=3`。科学的条件を変更したときは設定を保存し、比較確認・ADRの更新を行う。

背景上昇は実時刻による5秒基線・5秒上昇・5秒持続が暫定既定。詳細と診断は[背景上昇の手順](background-rise.md)、旧方式の再現は `--rise-method legacy` と `--baseline-ratio`、`--sigma-threshold`、`--min-consecutive` を使う。手動上昇時刻と解析範囲は従来どおりデータの `config.ini` で指定する。背景が未提供・境界が未確定なら前後解析の未実行を明示する。

元重心は `outputs/<day>/raw_centroid/<method>/<AVI名>.csv` に方式ごとに保存する。旧方式のキャッシュは上書きしない。新しい面積重心の抽出と、旧輪郭平均を固定した中心手法比較は異なる実行である。

## 同じ入力から再計算する

```bash
python3 scripts/04_repellent_response_analysis.py --day repellent-response/<dataset> \
  --measurement-input <local-input-manifest.json> --output-suffix reproduced
```

入力manifestの例（すべてのファイルパスはmanifestからの相対パスまたは絶対パス）:

```json
{
  "samples": [{
    "sample_no": 1,
    "centroids": "centroids.csv",
    "metadata": "centroids.json",
    "acquisition_tiff": "first-frame.tif",
    "background": "background.csv",
    "background_time_column": "No.1_time",
    "background_column": "No.1_background",
    "rise_source_frame": 100,
    "identity": "same source track assumed within each continuous observed run; unverified"
  }],
  "details": true,
  "zoom_ranges": [{"sample_no": 1, "start_s": 0.1, "end_s": 0.4, "label": "Center completion"}]
}
```

`centroids.csv` は `source_frame_1based,time_sec,x_um,y_um` を必須とする。元画素座標 `x_px,y_px` があればその座標で中心を推定し、μm座標との校正整合を検査する。`detected,reacquired,segment_id,target_id` がある場合は採否・境界を保持する。metadataは `sample_no,frame_count,method` と、元画素座標がある場合の `px2um_x,px2um_y` を保存する。

撮影周期と時刻精度はTIFFメタデータから読む。背景CSVは元重心の全時刻と一致させる。背景・上昇境界は省略可能だが、未指定なら前後解析は未実行とする。元の試料番号は保持し、選択した試料を再採番しない。入力ファイルのhash、入力重心の方式、中心設定を `settings.json` に保存する。入力を含む出力先への書き戻しは拒否するため、元結果を入力にする場合は別のsuffixを指定する。

この経路は中心・補完・角速度・SD・FFTを計算し直す。既計算結果の再描画だけの場合は `--results-bundle <bundle.json>` を使う。両者は同時指定できない。結果bundleの表は [出力writer](../../utils/functions/repellent_result_export.py) の契約に従う。

## 出力構成

```text
outputs/[suffix/]<day>/repellent_response/
├── index.html
├── settings.json
├── output_manifest.csv
├── sample_index_map.csv
├── 00_time_list/
│   ├── time_list.csv
│   └── analysis_frame_ranges.csv
├── 01_brightness_change/
│   ├── rise_summary.csv
│   ├── background_intensity_time_series.csv
│   └── background_intensity_time_series.png
├── 00_all_rotational_analysis/
│   ├── centroid_coordinate/  # 重心、欠測理由、XY・時系列図
│   ├── center_coordinate/    # 中心、推定採否・理由、補完状態、支持点
│   └── angular_velocity/     # 角速度表、全時間図（0線あり）
├── 02_pre_rise_fluctuation/fluctuation_analysis/
│   ├── SD_time_series.csv
│   ├── SD_time_series.png
│   ├── SD_fft.csv
│   ├── SD_fft_loglog.png
│   └── FFT_features.csv
├── 03_post_rise_analysis/switching_analysis/
│   ├── reversal_candidates.csv
│   ├── reversal_frequency.csv
│   └── ...                  # 候補・観測品質の図
└── details/                 # 有効時のみ：補正前中心・指定範囲の拡大
```

正確なファイル一覧は毎回 `output_manifest.csv` と `index.html` に保存する。試料別フォルダ・実行IDは作らない。表は全試料を集約し、図は `repellent_samples_per_page` に従って必要時のみページ分割する。

中心・重心・角速度図の `_labeled` は欠測理由・短補間・平均補完・片側保持・記録gapを表示する。同名で接尾辞のない図はラベルなし。位置はμm、角速度はrad/s。`details/` の有無は `repellent_details_enabled`、再計算manifestでは `details` で変更する。手法選択用の1/3/5点比較は含めない。

再実行は前回manifestに登録された出力だけを整理する。未知のファイルや元データを一括削除しない。過去の実行を別に残す場合は明示したsuffixを使う。実測図・座標・Wordはローカル専用とし、Git管理へ含めない。

## 互換実行

`--center-estimator v2|validated|legacy` は過去の計算との互換確認用。標準の採用手法ではない。V2内部は従来のCW正だが、集約出力への変換時に一度だけCCW正へ変換する。停止測定は別列で保存し、主系列の欠測へ補充しない。[互換測定列の契約](repellent-reference-output.md) を参照。

補完後の角速度は `angular_velocity_rad_s`、補完前の参照による診断値は `uncompleted_angular_velocity_rad_s` に保存する。後者をV2主系列や補完後の角速度と混同しない。
