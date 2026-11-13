# 重要な解析判断（ADR）

手法、符号・時刻・単位、補完、主要指標、標準出力の互換性に関わる重要な変更は、実装と同時にADRへ記録する。
日常的な修正・作業ログ・レビュー受領記録はADRに含めない。

| ADR | 判断 |
|---|---|
| [0001](0001-observation-identity.md) | 元観測の保持、平滑化なし、CCW正、重心キャッシュの分離 |
| [0002](0002-local-conic-center-completion.md) | 局所楕円モデルによる中心推定と3点平均補完 |
| [0003](0003-repellent-output-contract.md) | フェーズ別出力、背景前後解析、再計算と再描画の分離 |
| [0004](0004-execution-provenance.md) | clean GitによるCLI実行、入力・ブランチ・環境を含むローカル実行履歴 |
| [0009](0009-poster-long-axis-fluctuation.md) | ポスター解析の長軸角、揺らぎ、合成対照 |
| [0010](0010-poster-phase-and-nyquist-band.md) | ポスター解析の推定回転位相、長軸専用評価、Nyquist帯域 |
| [0011](0011-poster-fixed-band-ratio.md) | ポスター解析の共通周波数帯による強度比と候補細胞比較 |
| [0012](0012-poster-band-components-and-steady-simulation.md) | Low・Low−Highの追加と非周期角速度の合成条件 |
| [0013](0013-poster-angle-noise-controls.md) | 合成対照を長軸角の直接生成に統一し、座標・周回角を除く |
| [0014](0014-poster-one-over-f-amplitude-surrogates.md) | 0.5秒窓FFT振幅の1/f形フィットと乱数位相の逆FFT |
| [0015](0015-poster-spectrum-envelope-fit.md) | 0.5秒窓FFT振幅の可変指数包絡フィット（0014のフィットを置換） |

各ADRには状態、背景、決定、代替、影響・限界、検証方法を記載する。後日の方針変更は新ADRで置き換え、既存の判断を黙って書き換えない。
