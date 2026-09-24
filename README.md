# multi_sensor_calibration

EVS、RGB、Thermalカメラを、ROS 2 bagまたはMetavisionのイベントファイルから
オフラインでキャリブレーションするためのROS 2 Pythonパッケージです。

ハードウェアトリガーが使えない構成を前提に、次を扱います。

- 共通刺激からの時刻offsetと線形driftの推定
- 各カメラのcheckerboard intrinsic calibration
- 時刻補正後のpairwise extrinsic calibration
- 自動検出できない場合の手動extrinsic保存
- EVS event windowとtimestamp規約の固定・記録

## EVS入力

EVSには次の3系統を同じフレームmanifestへ正規化して扱います。

| `evs.source` | 入力 | 用途 |
|---|---|---|
| `metavision_file` | Metavision RAWまたはevent-HDF5 | ROSを通さない第一候補 |
| `ros_events` | rosbagの`event_camera_msgs/EventPacket` | `event_camera_py`で復号して再window化 |
| `ros_event_image` | rosbagの`sensor_msgs/Image` | 既存のwindow化結果をそのまま利用 |

`metavision_file`が対応するHDF5は、`metavision_file_to_hdf5`で作られる
**event HDF5**です。`generate_hdf5.py`で作られるprecomputed tensor HDF5とは
異なります。

Metavision SDKはRAWとevent-HDF5を`EventsIterator`で同様に読めます。
HDF5 eventの`/CD/events`は`x, y, p, t[us]`を持ちます。

## OpenEB E2V grayscale reconstruction

`generate-evs`の`representation`には、単純な`polarity`、`count`に加えて
OpenEB 5.2.0 Core MLの`e2v`を選択できます。E2Vは学習済みのrecurrent modelで
event volumeからgrayscale画像を再構築します。

E2VはDocker内で実行する前提です。JetPilotの`Dockerfile.silky_evcam`はOpenEBの
Python bindingsと`metavision_core_ml`をbuildし、OpenEB同梱checkpointを次に保持します。

```text
/opt/openeb/sdk/modules/core_ml/models/e2v.ckpt
```

ホストmacOSへのOpenEB/PyTorch導入は不要です。E2V関連moduleは`representation: e2v`
を選んだ場合にだけ遅延importされます。

Jetson/arm64ではOpenEBとSilkyEvCam driverによるevent recordingだけを使用し、
PyTorch E2V検証はDocker build時にスキップします。E2V grayscale reconstructionは
x86_64/amd64 Docker環境で実行します。x86_64 imageは`ros2 run`と同じsystem Pythonへ
CUDA 13.0版PyTorchを導入し、Docker build時に`torch.version.cuda`のmajor versionと
`EventToVideo`のimportを検証します。build中はGPU deviceへ接続しないため、
`torch.cuda.is_available()`はE2V実行時に確認されます。

E2V出力は「指定時刻までのeventをmodelへ入力した後の状態」なので、timestampは
window centerではなくwindow endです。`representation: e2v`を選ぶとCLIが
`timestamp_policy: end`を適用します。設定として明示する場合は次のようにします。

```yaml
evs:
  representation: e2v
  e2v:
    checkpoint_path: /opt/openeb/sdk/modules/core_ml/models/e2v.ckpt
    device: auto
    warmup_frames: 5
    normalize_num_stds: 6.0
  window:
    schedule: periodic
    period_us: 20000
    accumulation_us: 20000
    timestamp_policy: end
    drop_partial_windows: true
```

modelはstatefulなので、各出力間のevent sliceを重複なしで順番に入力します。最初の
`warmup_frames`枚はmodel stateが安定するまで生成だけ行い、manifestには保存しません。
`frames.csv`には実際にmodelへ入力した区間を`window_start_us`と`window_end_us`で
記録します。

## Event windowとtimestamp契約

生成時刻を`T_end`、蓄積時間を`dt`とすると、フレームに含めるイベントは常に
次の半開区間です。

```text
[T_end - dt, T_end)
```

これは`polarity`と`count`で使用するMetavision
`PeriodicFrameGenerationAlgorithm`相当の規約です。E2Vはrecurrent stateを維持するため、
最初の区間以降は直前の出力時刻から現在の出力時刻までを重複なく入力します。

1枚ごとに次の時刻をすべて`frames.csv`へ保存します。

- `window_start_us`
- `window_end_us`
- `generation_timestamp_us`：常にwindow end
- `representative_timestamp_us`：他カメラとの照合に使う時刻
- `event_mean_us`：実際に含まれたイベント時刻の平均
- `reference_timestamp_s`：暫定または補正後の共通timeline

既定の`timestamp_policy`は`center`です。通常カメラの露光区間との対応を考えると、
window endよりwindow centerの方が自然な代表時刻だからです。ただしSDK互換の
generation timestampは必ず別に残します。

### Window schedule

`periodic`はEVS時計上で一定周期に生成します。`period_us == accumulation_us`なら
各イベントを原則1フレームだけに含めるfull accumulationです。

`reference_aligned`は、時刻同期結果を逆変換し、RGBの各frame時刻をEVSの
`representative_timestamp`へ一致させてon-demand生成します。既定の`center`では
RGB frame時刻がEVS window centerへ一致し、window endはそこから蓄積時間の半分だけ
後になります。intrinsic用の独立EVS画像は`periodic`、RGBとのextrinsic用画像は
`reference_aligned`を基本とします。

## 時刻の扱い

完全な同期ではなく、次のモデルを推定します。

```text
reference_time =
    sensor_time
  + offset_at_anchor
  + drift * (sensor_time - anchor_sensor_time)
```

EVS event rate、RGB frame difference、Thermal frame differenceを共通の活動信号として
windowごとに相互相関を取り、offsetとdriftを求めます。撮影時には3センサーすべてで
見える対象を大きく動かしてください。Thermalにも見える加熱板や人の手と、RGB/EVSで
検出しやすい高コントラストedgeを組み合わせると安定します。

### RAW/HDF5の注意

EVSのsensor timestamp原点はカメラstream開始であり、RAW録画開始ではありません。
またMetavisionのRAW→HDF5変換は既定でtimestamp shiftingを適用し、そのshift値は
HDF5内に保存されません。

JetPilotのOpenEB driverはRAW開始時のROS時刻を
`*.raw.metadata.yaml`へ保存します。本ツールはこれを**暫定anchor**として使い、
視覚活動の相関で残るoffsetとdriftを再推定します。

RAWをHDF5へ変換しても、元RAWのsidecarを残して
`evs.sources.metavision_file.metadata_path`から参照してください。

### ROS eventsの注意

`event_camera_py`でEVT3を先頭から順に復号します。途中から復号するとEVT3のwrap状態が
不明になるため、bagの途中seekは行いません。最初のevent sensor timeと最初の
EventPacket headerまたはbag時刻を暫定anchorにします。

## インストール

ROS依存をaptで入れたうえで、`rosbags`とMetavision SDKを利用環境へ用意します。

```bash
sudo apt install \
  ros-${ROS_DISTRO}-event-camera-py \
  python3-numpy python3-opencv python3-yaml

python3 -m pip install rosbags

colcon build --packages-select multi_sensor_calibration
source install/setup.bash
```

RAW/HDF5入力にはMetavision SDK Python bindingsが必要です。

CLIはROS 2パッケージの実行ファイルとして起動します。

```bash
ros2 run multi_sensor_calibration multi-sensor-calibration --help
```

## 設定

JetPilot用の初期設定は次です。

```text
config/jetpilot_evs_rgb_thermal.yaml
```

checkerboardの`columns`と`rows`はsquare数ではなくinner corner数です。

## 基本ワークフロー

### LED同期GUI用データのexport

RGBとEVSの両方に映したLEDを使う場合、LEDを囲むROIをそれぞれ
`x,y,width,height`で指定して、ブラウザGUI用の時系列を生成できます。
RGBは各frameの固定scale輝度、EVSは指定幅ごとの正負event数を出力します。

```bash
ros2 run multi_sensor_calibration multi-sensor-calibration led-sync-export \
  --config config/rc_popout_evs_rgb.yaml \
  --bag /workspaces/record/calibration_bag \
  --evs-source metavision_file \
  --event-file /workspaces/record/openeb/calibration.raw \
  --rgb-roi 100,80,60,60 \
  --evs-roi 75,70,50,50 \
  --bin-ms 1 \
  --preview-fps 20 \
  --preview-window-s 12 \
  --output-dir result/led_sync
```

出力は次です。

- `rgb_led_signal.csv`: `t,rgb`
- `evs_led_signal.csv`: `t,pos,neg`
- `led_sync_data.json`: GUIへそのまま読み込める統合データ
- `preview/rgb/*.jpg`: 記録開始・終了付近のRGB確認画像
- `preview/evs/*.jpg`: 同じ区間の極性event蓄積画像

時刻`t`は、RGB bag時刻とRAW sidecarのclock anchorを使って同じsession相対時刻へ
変換されます。`*.raw.metadata.yaml`をRAWの隣に置いてください。ROIは各sensorの
画像座標なので、RGBとEVSで同じ数値である必要はありません。
previewはGUIでLED位置を確認するためのもので、同期時刻の推定には1 msのEVS信号を
使用します。previewが不要な処理では`--preview-fps 0`で無効化できます。

### 1. bagの時刻診断

```bash
ros2 run multi_sensor_calibration multi-sensor-calibration inspect-bag \
  --bag /workspaces/record/calibration_bag \
  --config config/jetpilot_evs_rgb_thermal.yaml \
  --output result/bag_info.yaml
```

各topicについて`header - bag timestamp`の範囲を確認します。時刻domainが不明な段階では
設定の`timestamp_source: bag`を維持します。

### 2. 時刻offset/drift推定

RAWまたはevent-HDF5を使う場合:

```bash
ros2 run multi_sensor_calibration multi-sensor-calibration time-sync \
  --config config/jetpilot_evs_rgb_thermal.yaml \
  --bag /workspaces/record/calibration_bag \
  --evs-source metavision_file \
  --event-file /workspaces/record/openeb/calibration.raw \
  --output result/time_sync.yaml
```

ROS eventsまたは既存event imageを使う場合:

```bash
ros2 run multi_sensor_calibration multi-sensor-calibration time-sync \
  --config config/jetpilot_evs_rgb_thermal.yaml \
  --bag /workspaces/record/calibration_bag \
  --evs-source ros_events \
  --output result/time_sync_ros_events.yaml
```

`--evs-source ros_event_image`も選択できます。

### 3. EVS画像生成

まずperiodic画像でEVS intrinsicを求めます。

```bash
ros2 run multi_sensor_calibration multi-sensor-calibration generate-evs \
  --config config/jetpilot_evs_rgb_thermal.yaml \
  --source metavision_file \
  --event-file /workspaces/record/openeb/calibration.raw \
  --output-dir result/evs_periodic
```

OpenEB E2Vでgrayscale画像を再構築する場合は次を実行します。E2V用の
`timestamp_policy: end`は自動的に適用されます。

```bash
ros2 run multi_sensor_calibration multi-sensor-calibration generate-evs \
  --config config/jetpilot_evs_rgb_thermal.yaml \
  --source metavision_file \
  --representation e2v \
  --event-file /workspaces/record/openeb/calibration.raw \
  --e2v-checkpoint /opt/openeb/sdk/modules/core_ml/models/e2v.ckpt \
  --e2v-device auto \
  --output-dir result/evs_e2v_periodic
```

extrinsic用には設定の`evs.window.schedule`を`reference_aligned`へ変更し、時刻同期結果と
RGB bagを渡します。

```bash
ros2 run multi_sensor_calibration multi-sensor-calibration generate-evs \
  --config config/jetpilot_evs_rgb_thermal.yaml \
  --source metavision_file \
  --event-file /workspaces/record/openeb/calibration.raw \
  --bag /workspaces/record/calibration_bag \
  --time-sync result/time_sync.yaml \
  --output-dir result/evs_rgb_aligned
```

E2Vでも同じ`reference_aligned` scheduleを使用できます。E2Vでは代表時刻がwindow end
なので、RGB frame時刻までのeventをmodelへ入力した直後の再構築画像が対応付けられます。

### 4. Intrinsic calibration

```bash
ros2 run multi_sensor_calibration multi-sensor-calibration intrinsics \
  --config config/jetpilot_evs_rgb_thermal.yaml \
  --sensor rgb \
  --bag /workspaces/record/calibration_bag \
  --output result/rgb_intrinsics.yaml

ros2 run multi_sensor_calibration multi-sensor-calibration intrinsics \
  --config config/jetpilot_evs_rgb_thermal.yaml \
  --sensor evs \
  --images-dir result/evs_periodic \
  --output result/evs_intrinsics.yaml
```

Thermalも`--sensor thermal --bag ...`で同様です。通常印刷checkerboardがThermalで
見えない場合は、材質差または加熱によって温度edgeを作る必要があります。

### 5. Pairwise extrinsic calibration

```bash
ros2 run multi_sensor_calibration multi-sensor-calibration extrinsics \
  --config config/jetpilot_evs_rgb_thermal.yaml \
  --bag /workspaces/record/calibration_bag \
  --reference rgb \
  --sensor evs \
  --sensor-images-dir result/evs_rgb_aligned \
  --reference-intrinsics result/rgb_intrinsics.yaml \
  --sensor-intrinsics result/evs_intrinsics.yaml \
  --time-sync result/time_sync.yaml \
  --output result/rgb_to_evs.yaml
```

RGB–Thermalも同様に実行します。

### 6. 手動extrinsic

自動検出が成立しない場合は、TFのparent→child poseを明示して保存できます。

```bash
ros2 run multi_sensor_calibration multi-sensor-calibration manual-extrinsic \
  --parent-frame realsense_color_optical_frame \
  --child-frame boson_optical_frame \
  --xyz 0.03 0.00 0.01 \
  --rpy 0.00 0.00 0.00 \
  --output result/rgb_to_thermal_manual.yaml
```

手動結果には`unvalidated`が記録されます。別データでのoverlay検証後にのみruntimeへ
反映してください。

## Kalibr pipeline

ROS 2/OpenEB側では、時計補正済みのmono8 PNGデータセットまでを生成します。
ROS 1 NoeticとKalibrは`tools/kalibr`の専用Dockerだけに閉じ込めます。

設定の`kalibr.cameras`はKalibr camera chainの順番です。隣接カメラに共通観測が必要なため、
3カメラでは`EVS → RGB → Thermal`を既定にしています。RGB＋EVSだけで検証する場合は、
使用する設定ファイルの`kalibr.cameras`からThermalを外してください。

E2V画像を先に`reference_aligned`で生成した後、Kalibr入力をexportします。

```bash
ros2 run multi_sensor_calibration multi-sensor-calibration export-kalibr \
  --config config/jetpilot_evs_rgb_thermal.yaml \
  --bag /workspaces/record/calibration_bag \
  --time-sync result/time_sync.yaml \
  --images-dir evs=result/evs_e2v_aligned \
  --output-dir result/kalibr_dataset
```

`--images-dir SENSOR=PATH`は複数回指定できます。指定されなかったcameraはROS 2 bagの
`image_topic`から読み出します。

export処理は次を保証します。

- `time_sync.yaml`の時計モデルを各timestampへ一度だけ適用
- RGB基準の約4 Hzの時刻列に最も近い画像を各cameraから一対一で選択
- `approximate_sync_s`を超える組と、全cameraが揃わない組を除外
- 歪み補正していないsingle-channel 8-bit PNGを生成
- 補正後timestampをナノ秒整数のファイル名として保存
- 入力timestamp、時計補正量、補正後timestamp、基準時刻との差を`manifest.csv`へ保存
- checkerboard設定をKalibr形式の`target.yaml`へ変換
- 使用した時計モデルを`time_sync.yaml`としてdataset内へ複製

生成物は次の構造です。

```text
kalibr_dataset/
├── cam0/<corrected_timestamp_ns>.png
├── cam1/<corrected_timestamp_ns>.png
├── cam2/<corrected_timestamp_ns>.png
├── manifest.csv
├── target.yaml
├── time_sync.yaml
└── job.yaml
```

Kalibr専用imageをbuildし、入力をread-only mountして実行します。

```bash
./tools/kalibr/build.sh

./tools/kalibr/calibrate.sh \
  result/kalibr_dataset \
  result/kalibr_output
```

Docker内では`kalibr_bagcreater`によるROS 1 bag生成と
`kalibr_calibrate_cameras`だけを行います。結果は`kalibr-camchain.yaml`、
詳細text、PDF report、実行log、使用したKalibr commitを記録したmetadataです。
既存結果の意図しない上書きを避けるため、datasetと出力ディレクトリは空である必要が
あります。

### EVS–RGB空間校正の重畳確認

RCカー飛び出し実験で使用する既定のD455 + SilkyEvCam空間校正は
`config/calibrations/rc_popout_default/`に保存しています。固定マウントを変更せず、
EVS 640x480・RGB 848x480の撮影条件を維持する場合に再利用できます。時刻同期は
セッションごとにLED記録から確認してください。

Kalibr結果と書き出し済みdatasetから、RGB画像上へEVSを半透明投影した確認動画を
生成できます。カメラ間には並進があるため、全深度へ通用する単一homographyは存在
しません。このコマンドは各RGB画像でcheckerboard姿勢を推定し、その平面上だけで
幾何学的に正しい射影を行います。

```bash
ros2 run multi_sensor_calibration multi-sensor-calibration calibration-overlay \
  --dataset result/kalibr_dataset \
  --camchain result/kalibr_output/kalibr-camchain.yaml \
  --alpha 0.45 \
  --output-dir result/calibration_overlay
```

Metavision RAWを指定すると、LEDで推定した`time_sync.yaml`を使って各RGB時刻へ
RAWイベントを対応付け、正極性を青、負極性を赤で追加表示できます。既定ではRGB時刻を
中心とした10 msを蓄積します。

```bash
ros2 run multi_sensor_calibration multi-sensor-calibration calibration-overlay \
  --dataset result/kalibr_dataset \
  --camchain result/kalibr_output/kalibr-camchain.yaml \
  --event-file recording/openeb_camera.raw \
  --event-window-ms 10 \
  --event-window-position center \
  --event-dilate-px 1 \
  --alpha 0.8 \
  --output-dir result/calibration_overlay_polarity
```

生成物:

- `overlay_blend.mp4`: RGBと色付きEVSグレースケールの半透明合成
- `overlay_edges.mp4`: RGB上へEVS edgeをcyanで重畳
- `overlay_polarity.mp4`: RGB上へRAWイベントを青（正）・赤（負）で重畳
- `polarity_only.mp4`: RAWイベントだけを白背景へ青（正）・赤（負）で描画
- `snapshots/`: 記録全体から抽出した確認用PNG
- `summary.yaml`: 検出数とcheckerboard cornerの整合誤差

`overlay_edges.mp4`では緑円がRGB corner、magenta十字がEVS cornerをRGBへ射影した
位置です。背景などcheckerboardと異なる奥行きの物体にはparallaxが残るため、評価は
checkerboard上の線とcornerで行います。

### 飛び出しシーケンスの全区間同期・重畳動画

checkerboardのない実験記録では、保存済み空間校正と各sessionのLED時刻補正を使い、
RAW極性eventをRGBへ重ねた全区間MP4を生成できます。既定の`rotation-only`はDSEC型で、
並進を無視するため奥行きによるparallaxは残りますが、点灯・消灯やRCカー運動の時間同期を
連続動画で確認できます。

```bash
ros2 run multi_sensor_calibration multi-sensor-calibration scenario-overlay \
  --bag /workspaces/record/evs-popup-v1/session \
  --event-file /workspaces/record/evs-popup-v1/session/openeb_camera.raw \
  --time-sync result/time_sync_led.yaml \
  --camchain config/calibrations/rc_popout_default/kalibr-camchain.yaml \
  --projection rotation-only \
  --event-window-ms 10 \
  --event-window-position center \
  --alpha 0.85 \
  --output-dir result/scenario_overlay
```

生成物は、RGB重畳の`overlay_polarity.mp4`、白背景eventの`polarity_only.mp4`、RGBと
重畳を横に並べた`rgb_vs_overlay.mp4`です。全区間が既定で、`--start-s`と
`--duration-s`を指定した場合だけ部分出力します。対象のおおよそのEVS奥行きが既知なら、
`--projection fixed-depth --depth-m 1.0`のように指定できます。この場合、指定した
fronto-parallel plane上だけが幾何学的に一致します。動画は生成後に既定で
H.264・yuv420p・fast-start MP4へ変換するため、macOS QuickTimeとbrowserで再生できます。
変換にはcontainerに導入済みのGStreamer `x264enc`を使います。デバッグ目的で従来の
OpenCV `mp4v`を残す場合だけ`--keep-opencv-mp4v`を指定してください。

### LED ROIと時刻同期の自動推定

`led-sync-export`が生成した16 px空間タイルから、開始・終了それぞれのRGB/EVS LED ROIを
32・48・64 pxのsliding windowで独立に探索できます。既知の2.3秒点滅周期と正負極性を
使い、単発の物体移動より反復点滅を優先します。

自動結果、開始・終了preview、生成済み重畳動画を目視確認用HTMLへまとめる場合は、
次を実行します。

```bash
ros2 run multi_sensor_calibration multi-sensor-calibration led-sync-review \
  --data-json result/led_sync_data.json \
  --auto-result result/auto_led_sync_result.json \
  --video-dir result/scenario_overlay \
  --output-dir result/review
```

`review/index.html`に開始・終了のROIデバッグ画像、confidence、対応edge数、coverage、
precision、candidate gapと、RGB/Event重畳動画が表示されます。

```bash
ros2 run multi_sensor_calibration multi-sensor-calibration auto-led-sync \
  --data-json result/led_sync_data.json \
  --output-yaml result/time_sync_led_auto.yaml \
  --output-json result/auto_led_sync_result.json
```

JSONには4つのROI、各候補の時刻検出信頼度と位置一意性、coverage、edge precision、
時計モデルと対応エッジを保存します。複数の場所にLED反射が現れても時刻同期品質を
不当にlowにしないため、`timing_confidence`と`localization_confidence`を分離しています。
またRGB観測区間とEVS binから求めた`timestamp_quantization_bound_s`、対応edgeの
`max_abs_residual_s`と`p95_abs_residual_s`も保存します。YAMLは`scenario-overlay`へ直接
指定できます。全記録を連続処理する場合は
JetPilot側の`scripts/experiments/run_rc_popout_auto_pipeline.sh`を使用します。

## 現時点の制約

- checkerboardとpinhole/radtan modelのみ
- 時刻同期はsoftware推定であり、hardware同時性を保証しない
- rolling shutter、Thermalの応答遅れ、通常カメラの露光timestamp semanticsは別途評価が必要
- ROS EventPacketの復号には`event_camera_py`が必要
- E2VにはDocker内のOpenEB Python bindings、`metavision_core_ml`、PyTorchが必要
- E2V checkpointはOpenEB 5.2.0同梱版を前提とし、信頼できないcheckpointを読み込まない
- 外部キャリブレーションは2カメラずつRGB基準で求める
- Kalibr DockerはROS 1 Noeticを使用し、ROS 2/OpenEBコンテナとは統合しない
- Kalibr exportはcamera間の時計推定を行わず、事前に生成した`time_sync.yaml`を必須とする
- 自動結果をJetPilot runtime設定へ自動反映しない

## 参考仕様

- [Metavision HDF5 Event File Format](https://docs.prophesee.ai/stable/data/file_formats/hdf5.html)
- [Metavision Timestamp Shifting](https://docs.prophesee.ai/stable/data/streaming_decoding/timestamp_shifting.html)
- [Metavision Generating Frames](https://docs.prophesee.ai/stable/guides/frames_generators.html)
- [event_camera_py](https://github.com/ros-event-camera/event_camera_py)
