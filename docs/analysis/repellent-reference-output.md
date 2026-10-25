# 忌避応答の最終参照中心と停止角速度

回転中のV2参照中心と、条件を満たす停止候補の補間中心を、最終参照中心として保存・表示する。
中心の定義、角速度の計算可否、計算値の品質は別に扱う。
V2の参照・符号、停止候補判定、前後参照、補間・品質の閾値、約0.2秒の評価窓と約0.025秒の間隔を維持する。
物理中心の位置精度や不確かさの校正は、この保存・描画機能の保証範囲に含めない。

## 時系列CSVの保存列

`rotation_center/noXX/timeseries.csv` は主系列と分離した次の測定列を保存する。
位置はµm、角速度はrad/s、CW正。行は既存の評価点に対応する。
`source_frame_1based` は入力AVIの元フレーム、`frame_0based` は選択範囲内のインデックス、
`segment_id` は座標・時刻が連続する記録を表す。

| 列 | 意味 |
| --- | --- |
| `final_reference_x_um`, `final_reference_y_um` | V2採用点ではV2中心、それ以外の補間可能な停止候補点では前後参照の線形補間中心 |
| `final_reference_source` | `v2` / `stop_interpolated` / `missing` |
| `final_reference_reason` | 採用または中心欠測の理由 |
| `stop_angular_velocity_raw_rad_s` | 計算可能な停止角速度。低信頼値も含む。0固定しない |
| `stop_velocity_quality` | `qualified` / `low_confidence` / `unavailable` |
| `stop_velocity_reasons` | 品質・計算不能の理由。複数理由はセミコロン区切り |

最終中心は解析用の参照であり、独立に観測された物理中心ではない。
中心の保存可否と、角速度の計算可否・品質を別に扱う。
補間可能な候補境界では中心が存在しても角速度は欠測となる。
欠測時刻・欠測座標・未撮影区間をまたいだ補間、片側参照での固定、評価点間の補充はしない。
8秒以下の候補、前後4秒以内の参照探索と支持条件を維持する。

### 品質と理由

| 理由 | 意味 |
| --- | --- |
| `v2_supported` | V2が採用した参照中心 |
| `stop_interpolated` | 前後参照から補間できる停止候補中心 |
| `qualified` | 停止角速度を計算でき、既存の品質条件にも適合 |
| `unresolved_lever_arm` | 中心との距離がノイズ・参照のばらつき・回転時の運動量に対して不足 |
| `inconsistent_radial_motion` | 評価窓内の半径変化が既存条件を超える |
| `candidate_boundary` | 約0.2秒の評価窓が停止候補の範囲に収まらない |
| `recording_boundary` | 評価窓が同じ連続記録に収まらない |
| `undefined_angle` | 補間中心と平滑化重心が数値精度の範囲で一致し、角度を定義できない |
| `nonfinite_velocity` | 計算結果が有限にならない |
| `missing_left_reference`, `missing_right_reference` | 規定条件を満たす前／後参照がない |
| `stop_too_long` | 停止候補が8秒を超える |
| `not_stop_candidate` | 停止測定の対象外 |
| `disabled` | 停止測定を無効にした実行 |

距離・半径変化の不合格のみなら、計算値を `low_confidence` として保存する。
品質理由は `unresolved_lever_arm`、`inconsistent_radial_motion` の順。
角度未定義の場合は `undefined_angle` をその前に付け、値は欠測にする。
中心がない通常点の `final_reference_reason` は既存V2の `decision_reason` を保持する。
V2判定不能を一律に停止候補とは扱わない。

既存の `measurement_center_*` と `stop_angular_velocity_rad_s` は品質適合時だけの測定列として保持する。
既存の `measurement_reason` は従来の最初の不合格理由を保持し、追加列の複数理由とは分ける。
主中心・主角速度、CW率・切替数・既存揺らぎ解析には追加した中心・停止値を使用しない。
`omega_noise_scale_rad_s` は既存の近似指標であり、校正済み信頼区間ではない。

`summary.csv` の `final_reference_{v2,stop_interpolated,missing}_{count,fraction}` は全評価点を分母とする。
`stop_quality_{qualified,low_confidence,unavailable}_{count,fraction}` は停止候補評価点を分母とする。
分母はそれぞれ `final_reference_denominator`、`stop_quality_denominator`。
分母0では件数0、割合NaN。空系列と停止測定無効時も追加列を保持する。
`settings.json` の出力スキーマ版は2。

## 図と刺激前後の分割

通常の `centroid_coordinate/` に以下を保存する。試料数が多い場合は従来のページ分割に従う。

- `centroid_before.png`：重心に最終中心を重ねる。V2と補間中心は同じ色。XY・x時系列・y時系列。
- `rotation_center.png`：同色の最終中心。XY・x時系列・y時系列。
- `centroid_before_components.png`、`rotation_center_components.png`：V2青、停止補間橙の内訳。
- `angular_velocity/angular_velocity_with_stop.png`：主角速度青、品質適合停止角速度は橙線、低信頼値は橙×。計算不能値は描かない。

全期間・刺激前・刺激後で同じ表を使う。
各期間の `rotation_center/noXX/timeseries.csv` は全期間の表を抽出したもので、値を再計算しない。
停止エピソード表はその期間に交差する元エピソードを保存し、描画帯を表示期間に切り詰める。
元フレーム・元時刻・状態・品質は分割前の値を保持する。
立ち上がりフレームが指定されている場合、重心と評価表は同じフレーム境界で分割する。
そのフレームの時刻が欠測でも前後の所属を保持し、欠測時刻自体は補完しない。

停止候補は薄橙帯、撮影・座標欠測は灰色帯。
時間図もXY図も記録区切り・未定義点をまたぐ線を結ばない。孤立点はマーカーで残す。
診断図も同じ最終中心・停止角速度を使用し、描画側で補間や採否判定を行わない。
従来の主系列だけの角速度図、補正重心、解析用中心CSVは主系列として残る。

## 実行と保存先

```bash
python3 scripts/04_repellent_response_analysis.py --day repellent-response/23
python3 scripts/04_repellent_response_analysis.py --day repellent-response/23 --all-frames
python3 scripts/04_repellent_response_analysis.py --day repellent-response/23 --all-frames --output-suffix verification
```

保存先は `outputs/[suffix/]<day>/repellent_response/runs/<configured|all_frames>/<run_id>/`。
実行IDは日本時間の日時と衝突防止ID。毎回新しいフォルダを作成し、冒頭にパスを表示する。
従来の保存済み成果物を自動移動・削除・上書きしない。入力AVI/TIFF、共有の `raw_centroid` は実行出力先から分離する。
既存の共有 `time_list.csv` は読み取り利用し、列内の欠測時刻を詰めない。
入力やキャッシュがない場合の既存の生成経路は維持する。

`run.json` は開始時 `running`、正常終了時 `success`、例外時 `failed`。
失敗時も途中成果物とエラー種別を残し、出力コンテキストを解除する。
解析モード・実行ID・使用原フレーム範囲は `run.json`、`analysis_method.json`、
`rotation_center/settings.json` と図へ記載する。
実行記録と設定JSONの `input_centroid_method` は入力生重心の抽出方式を記録する。
ここでの抽出方式v2/v3と、回転中心のV2推定モデルは別の名称である。
`00_time_list/analysis_frame_ranges.csv` は設定範囲と元時刻の対応。
No.14の通常解析は原フレーム2049–16384で、全フレーム検証の結果と区別する。

## 検証

```bash
python3 -m pytest -q
python3 -m black . --diff --check
python3 -m isort . --diff --check
python3 -m ruff check .
python3 -m mypy --python-version 3.10 scripts utils tests
```

回帰テストは中心保存と品質の分離、複数理由、候補境界、参照不足、長時間停止、
座標欠測・時刻ギャップ・孤立点、空系列・停止測定無効、出力分離・既存成果物保持・失敗記録、
図の実際の描画座標とCSV、刺激前後の元フレーム対応を確認する。

## 互換性の確認範囲と限界

保存済みの旧重心入力と面積重心v3入力を別々に確認した。
同じ入力方式と同じ原フレーム範囲で基準実装と比較し、実測14試料の主系列と既存の品質適合停止測定列が数値許容差内で一致した。
入力方式間の差は保存・描画機能の回帰とは扱わない。

面積重心v3の設定範囲解析は揺らぎ解析まで完走した。
全期間・刺激前・刺激後の42表について元フレーム・時刻・品質・保存値の対応を確認し、
CW率・切替数が主系列だけから計算されることを確認した。
No.14は設定範囲2049–16384と全フレームを別に検証した。
この全フレーム確認は他13試料の全フレーム解析を意味しない。
既存252ファイルのサイズ・更新時刻を保持した。

内部・末尾・全欠測の時刻列でも元の行を保持する。
生重心キャッシュの時刻照合では同じ位置のNaNを一致とし、有限時刻への置換や異なる時刻は拒否する。
生重心メタデータがない場合はAVIのフレーム数で時刻列の長さを確認し、
時刻表の初回生成直後も欠測を除去しない読み込みを用いる。

実測座標・画像・動画・再現可能な時系列・実測由来図・詳細な実行記録はローカル保管のみ。
図とCSVの一致や手法間一致を、物理中心・停止判定・角速度の真値精度の証明とは扱わない。
元TIFFからAVIへの拡大・切り出しの座標変換や対象同一性の独立検証も、この互換性確認には含めない。
