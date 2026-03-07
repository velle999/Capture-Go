# Go Game — Windows .exe Build Guide

## What you need in one folder

```
go_board.html   ← the game (already built)
main.py         ← desktop launcher
build.bat       ← build script
```

## Steps

### 1. Install Python (if you haven't)
Download from https://python.org — make sure to check **"Add Python to PATH"** during install.

### 2. Put all 3 files in the same folder

### 3. Double-click `build.bat`

It will automatically:
- Install `pywebview` and `pyinstaller` via pip
- Bundle everything into a single `.exe`
- Output to `dist\GoGame.exe`

The whole process takes **30–90 seconds**.

### 4. Run `dist\GoGame.exe`

That's it — one file, no install needed, copy it anywhere.

---

## How it works

```
GoGame.exe
  └── pywebview  (lightweight native Windows webview wrapper)
       └── go_board.html  (the Go game, bundled inside the exe)
```

PyWebView opens a proper Windows desktop window (using the system's built-in
Edge/WebView2 renderer) and loads the HTML game inside it. No browser required,
no Electron bloat (~5 MB vs ~150 MB).

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python not found` | Re-install Python, check "Add to PATH" |
| `pip install` fails | Run `pip install pywebview pyinstaller` manually |
| White/blank window | Install WebView2 runtime: https://go.microsoft.com/fwlink/p/?LinkId=2124703 |
| Antivirus flags exe | Normal for PyInstaller exes — add an exception |
| Want a custom icon | Add `--icon youricon.ico` to the pyinstaller command in build.bat |

---

## Optional: Add a custom icon

1. Get a `.ico` file (convert any PNG at https://convertio.co)
2. Edit `build.bat`, add `--icon "youricon.ico"` to the pyinstaller command
3. Re-run `build.bat`

## Optional: Smaller exe with --onedir

Change `--onefile` to `--onedir` in build.bat for faster startup
(produces a folder instead of a single file).
