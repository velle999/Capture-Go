"""
Go Game — Desktop Launcher
Wraps go_board.html in a native Windows window using pywebview.
Build to .exe with: build.bat
"""

import webview
import sys
import os

def get_html_path():
    """Works both when running as script and as PyInstaller .exe"""
    if getattr(sys, 'frozen', False):
        # Running as compiled .exe — files are in _MEIPASS temp dir
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, 'go_board.html')

if __name__ == '__main__':
    html_path = get_html_path()
    url = f'file:///{html_path.replace(os.sep, "/")}'

    window = webview.create_window(
        title='9×9 Go — MCTS AI',
        url=url,
        width=900,
        height=820,
        resizable=True,
        min_size=(700, 660),
    )

    webview.start(debug=False)
