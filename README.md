# プロジェクト名
　数理人材育成協会HRAM(Human Resource Association of Mathematics)主催の
AIエキスパート人材育成コースで実施したスピーチ動作の評価モデル研究の成果である。
本コースではロボティックス関連の高野研究室に所属している。
本研究では、TEDサイトのスピーチ動画から話者の骨格座標及び音声を抽出し、
それら時系列データをAIモデルで処理することによりスピーチ動作の良し悪し判定を行うものである。
コース期間は2024年6月から約1年間であり最終的に学会及び公聴会での発表が必須である。<br>

<img src="docs2/images/speech.jpg" alt="スピーチ動作4分類" width="480">

## プログラム及びデータ概要
<11_データセット作成><br>
11a_Movenet_mp4topnkp.py：mp4動画からmovenetで骨格座標を抽出する<br>
11b_pnkp+Audio.py：mp3から音声データを抽出しフレーム同期させる<br>
11c_10sec_samples.py：骨格座標と音声データのテーブルから特定条件の10秒間のデータを抽出する<br>
12a_XXsec_samples.py：骨格座標と音声データのテーブルから特定条件(*1)の連続秒数最大のデータを抽出する<br>
12b_1secSW_samples.py：一連連続データサンプルからスライディングウィンドウ(1秒)サンプルを作成<br>
12c_XsecSW_samples.py：一連連続データサンプルからスライディングウィンドウサンプルを作成<br>
13a_LeftWrist_fixed.py：左手首固定のMovenet骨格座標データ作成<br>
13b_RightWrist_fixed.py：右手首固定のMovenet骨格座標データ作成<br>
13c_BothWrists_fixed.py：両手首固定のMovenet骨格座標データ作成<br>
13d_AudioStrength_fixed.py：音声データ固定の時系列データ作成、骨格座標はそのまま<br>
<21_学習評価><br>
21_transformer_encoder.py：4値分類学習、評価<br>
<設定ファイル:data2><br>
conf.csv：動画mp4データのフレームレート<br>
label.csv：学習データラベル<br>
results.csv：テストデータラベル<br>

## 入力データセット
data_csv.7z：学習データ(train/test)、評価データ(eva)<br>

## 動作環境
- Windows / Python(Anaconda等) / OpenAI-APIなど

## 基本的な使い方
(0)ビデオデータmp4のダウンロード<br>
TED talkサイトからスピーチ動画をダウンロードして640x480の動画へリサイズする。<br>
※TED動画は非営利目的であれば無償で利用可能である。<br>
スピーチ動画mp4に対してMovenetモデルにより骨格座標時系列データを取得する。<br>
並行して同動画mp4に対して音声強度mp3を取得し上記骨格座標とフレーム同期させる。<br>

(1)コアサンプルの作成<br>
スピーチ動画毎の上記骨格座標＋音声強度時系列データから上半身10秒の<br>
コアサンプルデータを抽出する。サンプルデータ数は996。<br>

(2)サンプルデータ増強<br>
以下の2種類の方法によりサンプルデータの増強を図る。<br>
これらデータ増強によりデータ数は7054まで増加する。
- スライディングウィンドウ
- データ加工(非活性化) 


(3)4値分類学習<br>

(4)評価<br>

## 出力ファイルと保存先
<出力ファイル:data2><br>
predict_eva.csv：評価結果出力結果<br>

## ファイルサイズ


## 関連リンク
HRAM<br>
https://hram.or.jp/guidance/<br>
TEDトークサイト<br>
https://www.ted.com/talks<br>

## 注意事項



