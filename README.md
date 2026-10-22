# Real-Time Heart Rate from Webcam Video (rPPG)

> **Team project** — CS 6983, Northeastern University, 2025. My contribution: the real-time inference pipeline (`rppg_system.py`, `calibrated_inference.py`) — face detection, FFT-based heart-rate estimation and the regression calibration step — and the live GUI (`rppg_gui.py`). Teammates led dataset preparation and model training experiments.

Reads a person's pulse from ordinary webcam video. No contact sensor.

**Results:** 7.2 bpm mean absolute error on UBFC-rPPG; calibration cut error on unseen subjects by 58% (10.4 → 4.4 bpm).

## How it works

1. **Face detection** — Haar cascade locates the face in each frame; the forehead/cheek region is tracked as the ROI.
2. **Signal extraction** — mean green-channel intensity per frame forms a raw photoplethysmography signal; band-pass filtered to 0.7–4 Hz (42–240 bpm).
3. **Heart rate** — FFT on a sliding 15-second window; the dominant frequency is the pulse.
4. **Learned models** — PhysNet and RTrPPG 3D CNNs trained on UBFC-rPPG (`unified_train.py`, `training.ipynb`) predict the pulse waveform directly from the video volume.
5. **Calibration** — a regression layer (`calibration.json`) corrects per-subject bias; this is what produces the 58% improvement on new faces.
6. **Live GUI** — `rppg_gui.py` shows the webcam feed, the extracted signal and the running heart-rate estimate.

## Run

```bash
pip install torch opencv-python scipy numpy matplotlib
python rppg_gui.py                      # live webcam demo
python calibrated_inference.py --video path/to/clip.mp4
python ubfc_dataset_processor.py        # prepare UBFC-rPPG for training
python unified_train.py                 # train PhysNet / RTrPPG
```

`final_model.pth` is the trained checkpoint used by the demo.

## Files

| File | Purpose |
|---|---|
| `rppg_system.py` | Core pipeline: ROI tracking, signal extraction, FFT HR estimation, model definitions |
| `calibrated_inference.py` | Inference with per-subject calibration applied |
| `rppg_gui.py` | Real-time desktop GUI |
| `ubfc_dataset_processor.py` | UBFC-rPPG loader and preprocessing |
| `unified_train.py`, `training.ipynb` | Training for PhysNet / RTrPPG |
