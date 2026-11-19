# Data card: recorded CARLA drives

Format after Gebru et al., *Datasheets for Datasets* (CACM 2021), shortened.

## What it is

Six ego-vehicle drives recorded in CARLA 0.9.16 by `carla_tools/record_dataset.py` (defaults: 40 NPC vehicles,
60 pedestrians, synchronous mode at 20 Hz, every 2nd tick saved). Each drive is 600 frames at 10 Hz (60 s) from a front camera, 800×450, 90° FOV,
mounted 1.5 m forward and 1.5 m up.

| Sequence | Town | Weather | Frames | Use |
|---|---|---|---|---|
| `Town01_sunset` | Town01 | ClearSunset | 600 | train / val |
| `Town03_clear` | Town03 | ClearNoon | 600 | train / val; MiDaS scale calibration |
| `Town03_rain` | Town03 | MidRainyNoon | 600 | train / val |
| `Town04_cloudy` | Town04 | CloudyNoon | 600 | train / val |
| `Town05_wet` | Town05 | WetNoon | 600 | train / val |
| `Town02_test` | Town02 | WetCloudySunset | 600 | **test only, scored once** |

Per frame: RGB (`rgb/`), metric depth from CARLA's depth camera (`depth/`), lane-marking masks from CARLA's semantic camera (`lanes/`), and
`gt.jsonl` with every actor's 2-D box, class, 3-D distance and persistent actor ID (used for tracking metrics).
`meta.json` stores town, weather, camera intrinsics and mount.

## Splits and leakage control

- Split by **town**, not by frame: the test town's roads, buildings and spawn points never appear in training.
- Validation = the last 15 % of each training sequence (`perception.data.carla_to_yolo`), so consecutive
  near-duplicate frames do not straddle train and validation. This gives 2550 train and 450 val images in
  `data/carla_yolo/`.
- Thresholds, tracker choice and INT8 calibration use training or validation frames only (`build_trt.py`
  refuses validation images for calibration).

## Labels

Ground truth comes from the simulator, not human annotation: 3-D actor bounding boxes are projected into the image
for vehicles and pedestrians within 60 m, and a box is kept only if at least 10 % of its pixels have a depth
close to the actor's distance (occlusion test against the depth camera). Evaluation additionally drops boxes
below a minimum pixel height. Error sources: the occlusion test works at box level,
boxes around partly visible actors, and CARLA's own actor geometry. The YOLO export keeps a 10-class
BDD-style label map; evaluation collapses it to pedestrian / vehicle, the only classes CARLA labels reliably.

## Known gaps

- Five daytime weathers (Town03 recorded twice, clear and rain); night and fog are absent.
- Six towns only; Town02 shares CARLA's asset library with the training towns, so "unseen" means unseen
  layout, not unseen visual domain.
- Simulated pedestrians are fewer in kind and move more regularly than real ones.

## Distribution

To be released as GitHub Release assets with SHA-256 checksums in `scripts/assets.json` and fetched by
`python scripts/fetch_assets.py` (test drive) or `--all` (training drives). **Status 2026-11-19: the release and
manifest are not published yet**, so a clean clone cannot fetch the data until they are. CARLA content is under the
CARLA licence (code MIT, assets CC-BY).
