# RT-DETR Multi-Camera Person Detection

This project runs PaddleDetection RT-DETR inference on two video sources (`cam1` and `cam2`) and applies simple IoU-based per-camera tracking for detected persons.

## Prerequisites

- Python 3.10 or newer (3.10 is recommended where Paddle wheels match your platform)

### macOS (Apple Silicon or Intel)

- Homebrew

### Linux (Debian/Ubuntu)

- `python3`, `python3-venv`, and `python3-full` so `python3 -m venv` can create a complete environment (`sudo apt install python3-venv python3-full`)

## Setup

### macOS

1. Install Homebrew (if not already installed):

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

2. Install Python 3.10:

```bash
brew install python@3.10
```

Follow Homebrew’s “next steps” to put `python3.10` on your `PATH`, or invoke it explicitly when creating the venv.

### Linux

Use your distribution’s Python 3 packages; do **not** run `pip install python@3.10` (that is not a PyPI package and system Python is often “externally managed”). Ensure the venv module is installed:

```bash
sudo apt install python3-venv python3-full
```

### Virtual environment (all platforms)

3. Create and activate a virtual environment:

```bash
python3 -m venv rtdetr_env
source rtdetr_env/bin/activate
```

If `python3 -m venv` fails with a missing `bin/python3`, remove any partial `rtdetr_env` folder (`rm -rf rtdetr_env`), install `python3-venv` / `python3-full` as above, and retry.

4. Upgrade `pip` and install Python dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

5. Install PaddleDetection:

```bash
git clone https://github.com/PaddlePaddle/PaddleDetection.git
cd PaddleDetection
pip install -r requirements.txt
python setup.py install
cd ..
```

6. Download RT-DETR weights into the project root:

```bash
wget https://paddledet.bj.bcebos.com/models/rtdetr_r50vd_6x_coco.pdparams
```

## Input Videos

Place your videos in the project root. The script looks for:

- `cam1.mp4` (fallback: `left.mp4`)
- `cam2.mp4` (fallback: `right.mp4`)

## Run

```bash
python main.py
```

Press `Esc` to stop.
