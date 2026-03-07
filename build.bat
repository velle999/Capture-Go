@echo off
setlocal
title Go Game — Build Script

echo.
echo ============================================
echo   Go Game .exe Builder
echo ============================================
echo.

:: ── Check Python ────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install from https://python.org
    pause & exit /b 1
)
echo [OK] Python found

:: ── Install dependencies ─────────────────────────────────
echo.
echo [1/3] Installing dependencies...
pip install pywebview pyinstaller --quiet
if errorlevel 1 (
    echo [ERROR] pip install failed.
    pause & exit /b 1
)
echo [OK] Dependencies installed

:: ── Check go_board.html exists ──────────────────────────
if not exist "go_board.html" (
    echo [ERROR] go_board.html not found.
    echo         Make sure go_board.html and build.bat are in the same folder.
    pause & exit /b 1
)
echo [OK] go_board.html found

:: ── Run PyInstaller ──────────────────────────────────────
echo.
echo [2/3] Building .exe with PyInstaller...
echo       (This takes 30-90 seconds)
echo.

pyinstaller ^
    --onefile ^
    --windowed ^
    --name "GoGame" ^
    --add-data "go_board.html;." ^
    --hidden-import "webview" ^
    --hidden-import "webview.platforms.winforms" ^
    --collect-all "webview" ^
    main.py

if errorlevel 1 (
    echo.
    echo [ERROR] PyInstaller failed. See output above.
    pause & exit /b 1
)

:: ── Done ─────────────────────────────────────────────────
echo.
echo [3/3] Done!
echo.
echo ============================================
echo   Output: dist\GoGame.exe
echo ============================================
echo.
echo Double-click dist\GoGame.exe to play.
echo You can copy GoGame.exe anywhere — no install needed.
echo.
pause
