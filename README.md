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

## Event windowとtimestamp契約

生成時刻を`T_end`、蓄積時間を`dt`とすると、フレームに含めるイベントは常に
次の半開区間です。

```text
[T_end - dt, T_end)
```

これはMetavision `PeriodicFrameGenerationAlgorithm`の規約と同じです。

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

## 現時点の制約

- checkerboardとpinhole/radtan modelのみ
- 時刻同期はsoftware推定であり、hardware同時性を保証しない
- rolling shutter、Thermalの応答遅れ、通常カメラの露光timestamp semanticsは別途評価が必要
- ROS EventPacketの復号には`event_camera_py`が必要
- 外部キャリブレーションは2カメラずつRGB基準で求める
- 自動結果をJetPilot runtime設定へ自動反映しない

## 参考仕様

- [Metavision HDF5 Event File Format](https://docs.prophesee.ai/stable/data/file_formats/hdf5.html)
- [Metavision Timestamp Shifting](https://docs.prophesee.ai/stable/data/streaming_decoding/timestamp_shifting.html)
- [Metavision Generating Frames](https://docs.prophesee.ai/stable/guides/frames_generators.html)
- [event_camera_py](https://github.com/ros-event-camera/event_camera_py)
