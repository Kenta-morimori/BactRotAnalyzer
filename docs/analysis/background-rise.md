# 背景Meanの持続上昇開始

[ADR 0009](../adr/0009-sustained-background-rise.md)の暫定検出条件を使う。
ユーザー作成Mean CSVと元TIFFの行対応は[画像入力手順](repellent-image-input.md)に従う。
背景上昇は刺激到達・回転反転の確定時刻ではない。

```bash
python scripts/04_repellent_response_analysis.py --day repellent-response/<dataset>
# 従来検出器の再現が必要な比較だけで明示する
python scripts/04_repellent_response_analysis.py --day repellent-response/<dataset> --rise-method legacy --output-suffix legacy-rise
```

既定は `sustained`。前5秒の基線、後5秒の区分線形フィットと追加5秒の持続を要求する。
Huber損失、6倍ノイズの上昇、単一直線から30%以上の改善、確認時間80%以上の上昇を使う。
探索刻みは0.25秒。観測を補間せず元フレーム・時刻へ対応付けるため、表示桁数を検出精度と読まない。
元のMean・重心・角速度は平滑化しない。欠測/gapを跨ぐ支持は作らない。

画像入力JSONの各 `background` には次を設定できる（数値は合成例）。

```json
{
  "csv": "brightness.csv",
  "format": "imagej_mean",
  "rise_detection": "auto",
  "rise_method": "sustained",
  "rise_parameters": {"baseline_sec": 5.0, "rise_sec": 5.0, "persistence_sec": 5.0}
}
```

`rise_parameters` の省略項目は `RiseSettings` の既定を使う。実効値を結果へ保存する。
窓や閾値は科学的な測定条件なので、結果に合わせて個別No.ごとに最適化しない。
CLI `--rise-method` があればJSONの方式より優先する。
`--baseline-ratio`、`--sigma-threshold`、`--min-consecutive` は明示 `legacy` 用で、新法の閾値を変更しない。
値だけのPython APIでも従来法を再現するときは `method="legacy"` を指定する。
新法は `times=` を必須の支持とし、必要に応じ `source_frames=`・`segment_ids=`・`max_dt=` を渡す。

`rise_source_frame` がある場合は方式・自動パラメータより優先する。
既存の手動境界を自動法で置き換えない。手動境界は元時刻のある既存行を指す必要がある。
`--measurement-input` と `--results-bundle` は保存済み境界を使い、再検出しない。

## 診断の読み方

`settings.json` の `image_input.rise_results` に方法、自動検出器、実効設定、理由、候補、コードhashを保存する。
`01_brightness_change/rise_summary.csv` に方式・状態・検出器・理由・候補数を表示する。
従来のday/config入口でも `settings.json` の `rise_results` と `rise_detection` を確認する。

| 理由 | 解釈・次に確認すること |
|---|---|
| `sustained_rise` | 一つの候補群を採用。元波形・支持範囲を確認する |
| `multiple_candidates` | 離れた複数候補。境界は未判定で、自動的に最大スコアを選ばない |
| `insufficient_continuous_baseline_and_followup` | 基線と後続を支持する連続記録が不足 |
| `gap_or_missing_observations` | 欠測/gapで切った連続区間に採用候補がない。gap内の開始を補完しない |
| `missing_background_or_time` / `original_timestamps_required` | 必要な元観測・実時刻がない |
| `no_sustained_rise` | 条件を満たす候補なし。上昇なしと、端点・弱い/曲線的変化などの検出不能を区別する保証はない |
| `manual_source_frame` | 手動元フレームを優先した |

未判定はJSON null・CSV空欄/NaNのままとし、0秒にしない。前後フェーズの未実行を確認する。
候補の `candidate_time_range_sec` は採用探索位置の幅で、統計的信頼区間ではない。
非常に緩い上昇・長い一時上昇・非線形ドリフト・多段の曖昧さは残る。必要なら目視の不確かさを記録して手動境界を使う。

## 既存画像承認の再利用

背景の方式・設定・手動境界だけの変更では、承認済み画像測定・動画を再生成しない。
既知の旧全体ハッシュの成果物は、監査済み画像コードのAST・他の依存・環境・元画像・選択条件・成果物が全て一致するときだけ読み取り専用で再利用する。
既存キャッシュのCSV・メタデータ・承認JSONへ新hashを書き込まない。
未知の旧実装や画像条件の不一致は従来どおり再生成・承認を要求する。
`--force-extract` は明示的な再抽出なので再利用を抑止する。

実測の詳細設定・候補時刻・系列・図はローカル保管する。目視参照への一致を検出器の独立した精度保証にしない。
