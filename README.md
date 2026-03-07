# 囲碁 go-mcts-cuda

> 9×9 Go engine with CUDA-accelerated MCTS AI — playable in browser or as a Python/Pygame desktop app

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![CUDA](https://img.shields.io/badge/CUDA-12.x-green?logo=nvidia)
![License](https://img.shields.io/badge/license-MIT-brightgreen)

---

## Features

- **Full Go rules** — captures, ko, suicide prevention, Chinese area scoring with komi
- **MCTS AI** — Monte Carlo Tree Search with UCB1 selection
- **CUDA acceleration** — CuPy GPU rollouts via `cupy-cuda12x`, auto-falls back to CPU
- **Three ways to play** — browser UI, Python terminal, or Pygame GUI
- **Windows `.exe` builder** — one-click PyWebView + PyInstaller packaging
- **Built-in CUDA diagnostic** — `--cuda-test` flag checks your GPU, DLLs, bandwidth, and rollout speed

---

## Project Structure

```
go-mcts-cuda/
├── go_engine.py      # Python engine — MCTS AI, full Go rules, CUDA rollouts
├── go_board.html     # Browser UI — playable board with dev panel & benchmarks
├── main.py           # PyWebView desktop launcher
├── build.bat         # Windows .exe build script (PyInstaller)
└── README.md
```

---

## Quickstart

### Browser (no install)
Just open `go_board.html` in any modern browser. The full game engine runs in JavaScript — no server needed.

### Python terminal
```bash
pip install numpy
python go_engine.py
```

### Pygame GUI
```bash
pip install numpy pygame
python go_engine.py --pygame
```

### With CUDA (GPU acceleration)
```bash
pip install numpy pygame cupy-cuda12x
python go_engine.py --pygame
```

> **Note:** Install `cupy-cuda12x` for CUDA 12.x. Match the package to your **CUDA Runtime** version (not nvcc). Check with `python -c "import cupy; print(cupy.cuda.runtime.runtimeGetVersion())"`.

---

## Board Sizes

Pass `--size` to choose your board:

```bash
python go_engine.py --size 9     # default, fast, great for AI
python go_engine.py --size 13
python go_engine.py --size 19 --pygame
```

The browser UI has a 9×13×19 toggle button in the side panel.

---

## CUDA Diagnostic

Run a full GPU health check:

```bash
python go_engine.py --cuda-test
```

This reports:
- CuPy version and CUDA Runtime version
- nvcc compiler version and mismatch warnings
- GPU name, memory, SM count, compute capability
- DLL availability check (Windows)
- Host↔GPU bandwidth
- CPU vs CUDA rollout speed comparison

### Common Windows fix

If you see `DLL load failed while importing cublas`:

```powershell
# Temporary fix (current session only)
$env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin"

# Permanent fix — add this to your System PATH:
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin
```

---

## How the AI Works

The AI uses **Monte Carlo Tree Search (MCTS)** with UCB1 exploration:

```
Select        → walk tree using UCB1 score (exploitation + exploration)
Expand        → add one new child node
Rollout       → play random moves to end of game
Backpropagate → update win/visit counts up the tree
```

On each AI turn, it runs N simulations (configurable via `--sims` or the in-browser slider), then picks the move with the most visits.

**CUDA speedup:** rollouts pre-generate random move indices in bulk on the GPU via `cp.random.random()`, avoiding per-move Python overhead. This gives a meaningful speedup at higher simulation counts and larger board sizes.

---

## Windows .exe

Build a standalone `.exe` using PyWebView + PyInstaller:

```
# Put these files in the same folder:
go_board.html
main.py
build.bat

# Then double-click build.bat
# Output: dist\GoGame.exe (~5 MB, no install needed)
```

Requires Python and pip. The script installs `pywebview` and `pyinstaller` automatically.

> If the window is blank on Windows 10, install the [WebView2 Runtime](https://go.microsoft.com/fwlink/p/?LinkId=2124703).

---

## Requirements

| Package | Purpose | Install |
|---|---|---|
| `numpy` | Board arrays, CPU fallback | `pip install numpy` |
| `pygame` | Desktop GUI | `pip install pygame` |
| `cupy-cuda12x` | CUDA GPU rollouts | `pip install cupy-cuda12x` |
| `pywebview` | Desktop HTML wrapper | `pip install pywebview` |
| `pyinstaller` | `.exe` builder | `pip install pyinstaller` |

Python 3.10+ recommended.

---

## Dev Panel (Browser)

Click **⚙ Dev Panel** at the bottom of `go_board.html` for:

- **CUDA Diagnostics** — WebGL GPU renderer, hardware info, rollout speed test
- **Benchmark** — move generation, scoring, rollout, and MCTS simulation rates
- **MCTS Stats** — live per-move stats: sims, time, win%, rollout depth
- **Console** — intercepts `console.log/warn/error` from the game engine

---

## License

MIT
