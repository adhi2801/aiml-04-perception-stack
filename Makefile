# One-word entry points. Everything except `scenarios`/`falsify` runs without CARLA, on recorded drives.
#   make setup      venv + pinned deps (CUDA 12.4 wheels)
#   make assets     released weights + the Town02 test drive (SHA-256 checked)
#   make demo       full stack on the recorded test drive -> reports/demo_town02.mp4 + .gif
#   make eval       detection, tracking, lanes, depth on the unseen town -> reports/*.json
#   make results    regenerate the README tables from reports/*.json
#   make check      lint + format check + README drift gate + unit tests (what CI runs)
#   make scenarios  Euro NCAP-style closed loop in CARLA (CARLA server must be running)
#   make falsify    STL-robustness-guided scenario search + root-cause replay (CARLA)
#   make docker     CPU image that runs `make demo` / `make check` without a local Python setup

PY      ?= .venv/bin/python
SEQ     ?= data/carla/Town02_test
WEIGHTS ?= models/yolo11s_carla.pt
CFG     ?= configs/pipeline_lowvram.yaml
ifeq ($(OS),Windows_NT)
PY      := .venv/Scripts/python
endif

.PHONY: setup assets demo eval results check lint test scenarios falsify docker docker-demo

setup:
	python -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
	$(PY) -m pip install -r requirements.txt -c constraints-cu124.txt
	$(PY) -m pip install --no-deps -e .

assets:
	$(PY) scripts/fetch_assets.py

demo:
	$(PY) -m perception.demo --seq $(SEQ) --out reports/demo_town02.mp4 --gif reports/demo_town02.gif

eval:
	$(PY) -m perception.eval_detection --seq $(SEQ) --model coco=yolo11s.pt --model carla=$(WEIGHTS)
	$(PY) -m perception.eval_tracking --seq $(SEQ) --weights $(WEIGHTS)
	$(PY) -m perception.lanes.evaluate --seq $(SEQ) --weights models/lane_lraspp.pt
	$(PY) -m perception.depth.evaluate --calib data/carla/Town03_clear --test $(SEQ)

results:
	$(PY) scripts/results_table.py

lint:
	$(PY) -m ruff check src carla_tools tests scripts ros2_ws/src
	$(PY) -m ruff format --check src carla_tools tests scripts ros2_ws/src
	$(PY) scripts/results_table.py --check

test:
	$(PY) -m pytest -q

check: lint test

scenarios:
	$(PY) carla_tools/run_scenarios.py --scenario cpna25 --speeds 20,30,40 --trials 3 --config $(CFG) --weights $(WEIGHTS)
	$(PY) carla_tools/run_scenarios.py --scenario cpfa50 --speeds 20,30,40 --trials 3 --config $(CFG) --weights $(WEIGHTS)
	$(PY) carla_tools/run_scenarios.py --scenario ccrb --speeds 30,40 --trials 3 --config $(CFG) --weights $(WEIGHTS)

falsify:
	$(PY) carla_tools/falsify_carla.py --budget 30 --config $(CFG) --weights $(WEIGHTS)
	$(PY) carla_tools/root_cause.py --config $(CFG) --weights $(WEIGHTS)

docker:
	docker build -t perception-stack:cpu .

docker-demo: docker
	docker run --rm -v "$(CURDIR)/models:/app/models" -v "$(CURDIR)/data:/app/data" -v "$(CURDIR)/reports:/app/reports" \
		perception-stack:cpu make demo PY=python
