# 最終参照中心と停止角速度の保存・描画（#50 → #51）

親Issueは [#37](https://github.com/Kenta-morimori/BactRotAnalyzer/issues/37)。
保存契約を先に整備し、同じ表を使う通常図と診断図を追加する。
V2の参照・符号、停止候補判定、前後参照、補間・品質の閾値、約0.2秒の評価窓と約0.025秒の間隔は変更しない。
物理中心の精度改善と不確かさの校正は #46 に残す。

## 時系列CSVの追加列

`rotation_center/noXX/timeseries.csv` の既存列を保持し、次を追加する。
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

- `centroid_before.png`：重心と同色の最終中心。XY・x時系列・y時系列。
- `rotation_center.png`：同色の最終中心。XY・x時系列・y時系列。
- `centroid_before_components.png`、`rotation_center_components.png`：V2青、停止補間橙の内訳。
- `angular_velocity/angular_velocity_with_stop.png`：主角速度青、品質適合停止角速度は橙線、低信頼値は橙×。計算不能値は描かない。

全期間・刺激前・刺激後で同じ表を使う。
各期間の `rotation_center/noXX/timeseries.csv` は全期間の表を抽出したもので、値を再計算しない。
停止エピソード表はその期間に交差する元エピソードを保存し、描画帯を表示期間に切り詰める。
元フレーム・元時刻・状態・品質は分割前の値を保持する。

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

新規テストは中心保存と品質の分離、複数理由、候補境界、参照不足、長時間停止、
座標欠測・時刻ギャップ・孤立点、空系列・停止測定無効、出力分離・既存成果物保持・失敗記録、
図の実際の描画座標とCSV、刺激前後の元フレーム対応を確認する。

実測は最初に保存済みv2重心を固定入力として検証し、PR #53のマージ後にv3面積重心を別キャッシュへ抽出して再検証した。比較の前後で同じ重心方式を使用する。
実測座標・画像・動画・再現可能な時系列・実測由来図はローカル保管のみ。
実測の図とCSVの一致、手法間一致を物理中心や角速度の真値精度とは扱わない。

### 2026-10-23の初回確認結果（v2重心）

- 全pytest 104件が通過。Black（環境制限により57ソースをファイル単位で確認）、isort、Ruff、Python 3.10指定のmypyも通過。
- 変更前コード `20462ba` と9種類の模擬ケースを比較し、既存列・主中心・停止エピソードが一致した。
- 保存済みv2重心の実測14試料について、設定範囲で変更前コードを再計算し、主系列と既存の品質適合停止測定列が数値許容差内で一致した。
- 全評価点50,576点の内訳はV2中心42,785点、停止補間中心850点、中心欠測6,941点。停止候補2,047点の角速度は品質適合732点、低信頼14点、計算不能1,301点。補間中心がある角速度計算不能点104点も中心を保持した。
- 通常解析は揺らぎ解析を含めて終了コード0で完走し、148図を保存した。全期間・刺激前・刺激後の42表が同じ保存値を保持し、CW率・切替数が主系列だけから計算されることを確認した。従来の252ファイルはサイズ・更新時刻とも変わらなかった。
- No.14の全フレーム検証を通常解析とは別領域で実施し、変更前コードの既存列との一致を確認した。この全フレーム確認はNo.14を対象とし、他13試料の全フレーム再実行を意味しない。
- 初回の実測確認で設定メタデータのキー重複を検出して修正し、回帰テストを追加した。再実行では初回に保存した輝度CSVを元フレーム範囲に沿って再利用し、生重心は読み取り専用で固定した。

個別の実測表・図・実行記録・比較結果は、ローカルの
`outputs/issue50-51-validation/repellent-response/23/repellent_response/runs/` に保存する。


### PR #53マージ後の再確認結果（v3面積重心）

- devをマージコミット `70667f34e053c96a5e2f39e1e436b8a26828bb31` へ早送りした。既存の#50/#51の作業を保全し、#53の5ファイルを重複差分から除いた。
- v3入力は `outputs/issue50-51-v3-inputs/repellent-response/23/raw_centroid/` へ14試料を別途抽出し、v2入力と既存成果物を保持した。方式は `standard-largest-contour-area-centroid-threshold-120-v3`。
- 同じv3入力に対し、変更前コード `70667f3` と今回の実装の主系列・既存の品質適合停止測定列が14試料すべてで数値許容差内に一致した。v2入力とv3入力の差を#50/#51の回帰とは扱わない。
- 設定範囲の通常解析が揺らぎ解析を含めて終了コード0で完走した。161図と全期間・刺激前・刺激後の42表を確認し、元フレーム・品質・保存値の一致と、CW率・切替数が主系列だけから計算されることを確認した。
- 全評価点50,576点はV2中心43,245点、停止補間中心368点、中心欠測6,963点。停止候補1,199点の角速度は品質適合222点、低信頼26点、計算不能951点。この件数は保存契約の確認であり、物理中心や停止判定の精度を示さない。
- No.14の全フレームも別領域で確認し、同じv3入力を使った変更前コードの既存列と一致した。単独出力のCSV・図・実行記録でも元試料番号14を維持した。
- 通常解析のNo.14は原フレーム2049–16384を使用し、全フレーム検証と区別した。既存252ファイルはサイズ・更新時刻とも変わらなかった。
- 全pytest 106件、Black/isort/Ruff、Python 3.10指定mypyが通過した。片方の座標だけが欠測の場合も両成分の線を分割することを追加確認した。

v3の個別結果はローカルの
`outputs/issue50-51-v3-validation/repellent-response/23/repellent_response/runs/`、
入力方式とマージコミットの記録は `outputs/issue50-51-v3-inputs/input_provenance.json` に保存する。
v2の検証結果も別領域に残し、v3に置き換えない。

### 2026-10-24のPR仕上げ確認

- ローカルレビューで欠測時刻を含む生重心キャッシュの照合を修正した。同じ位置のNaNを一致として扱い、有限時刻への置換や異なる時刻は引き続き拒否する。
- 生重心のメタデータがまだない場合もAVIのフレーム数を使い、時刻列の末尾欠測・全欠測を保持する。時刻表を新しく生成した直後も、欠測を除去しない読み込みを使用する。
- 通常・内部欠測・末尾欠測・全欠測の生重心キャッシュ再利用と、キャッシュ未生成時の原フレーム数保持を回帰テストに追加した。これらは入力対応の修正であり、中心モデル・停止判定・品質閾値・評価窓を変更しない。
- 全pytest 112件、Black/isort/Ruff、Python 3.10指定mypyが通過した。ローカルの整形・静的検査は実験出力を含めず57ソースを対象とした。
