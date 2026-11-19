# Model card: perception stack (YOLO11-s CARLA detector, LR-ASPP lanes, MiDaS distance, TTC AEB)

Format after Mitchell et al., *Model Cards for Model Reporting* (FAT* 2019). Numbers are not repeated here:
they live in the README results tables, which `scripts/results_table.py` generates from `reports/*.json`
(CI fails if they drift). Each section below names the report that holds its numbers.

## Components

| Component | Model | Weights | Trained on |
|---|---|---|---|
| Detector | YOLO11-s (Ultralytics 8.4), COCO-pretrained, fine-tuned 40 epochs, AdamW, 640 px, batch 6, seed 0 | `models/yolo11s_carla.pt` (+ `.onnx`, TensorRT FP16 `.engine`, INT8 MinMax `_int8_minmax.engine`) | CARLA Town01/03/04/05 drives (see [DATA_CARD](DATA_CARD.md)) |
| Tracker | BoT-SORT without camera-motion compensation (no learned weights) | `configs/trackers/` | chosen on validation frames, `reports/tracker_choice.json` |
| Lanes | LR-ASPP MobileNetV3 binary lane segmentation, compared against a classical OpenCV pipeline | `models/lane_lraspp.pt` | CARLA training drives, lane masks from the simulator |
| Distance | Flat-ground camera geometry (default); MiDaS small calibrated to metres as the alternative | none / torch.hub | MiDaS scale fit on Town03_clear only |
| Decision | Time-to-collision AEB with ego-lane gating, hysteresis and a perception watchdog (rule-based) | none | thresholds in `configs/pipeline*.yaml` |

## Intended use

Research and teaching: a complete, measured camera-only AEB loop in simulation, to study where perception
errors turn into missed or late braking. **Not** for any real vehicle. The detector has never seen a real image;
nothing here has been validated outside CARLA 0.9.16.

## Evaluation protocol

- Test town (Town02, WetCloudySunset) is never used for training, threshold or tracker choice. It is scored once.
- Operating confidence chosen on validation frames from the training towns (`reports/operating_points.json`).
- Detection: COCO-style mAP50-95 / mAP50 with 95 % bootstrap CI over frames, pedestrian recall by distance band
  (`reports/detection_eval.json`). Classes collapsed to pedestrian / vehicle so the COCO baseline is comparable.
- Runtime: PyTorch / ONNX Runtime / TensorRT FP16 / INT8 (entropy vs MinMax, calibrated on training images only),
  p50/p95 latency and accuracy cost on the validation split, GTX 1650 4 GB (`reports/benchmark.json`,
  per-class cost in `reports/precision_report.json`).
- Tracking: MOTA, IDF1, ID switches vs CARLA actor IDs (`reports/tracking_*.json`).
- Lanes and distance: `reports/lanes_comparison_*.json`, `reports/depth_eval.json`.
- Closed loop: Euro NCAP-style CPNA-25, CPFA-50, CCRb at 20-40 km/h in CARLA, with oracle-detection and
  oracle-depth swaps for root cause (`carla_tools/run_scenarios.py`, results written to `reports/`).

## Known limitations (measured, not hypothetical)

- **Generalisation gap:** the pedestrian recall within 30 m that the operating confidence achieves on validation
  frames drops sharply on the unseen town (README, Detection section). Validation from training towns overstates
  real performance.
- **Distant pedestrians:** the fine-tuned detector recalls almost no pedestrians beyond 30 m at its operating
  confidence; the COCO model does better there. AEB at higher speed depends on exactly this range.
- **INT8:** entropy calibration costs several mAP points, concentrated in pedestrians; MinMax recovers most of it.
  FP16 is the safe deployment choice; INT8 MinMax is the fast one (benchmark tables).
- **Simulation only:** CARLA pedestrians, lighting and sensor noise are simpler than real roads. No sim-to-real
  evaluation is included.
- **Single camera:** no radar or lidar; distance is geometric or monocular and degrades on slopes and with range.

## Ethical and safety considerations

A missed pedestrian is the failure that matters, so the operating point is set by pedestrian recall, not mAP.
Results are reported per distance band and per scenario so that a good average cannot hide a dangerous case.
