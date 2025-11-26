@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
    python -m venv .venv
)
call .\.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
start "" http://127.0.0.1:8501
python -m streamlit run app_visual.py --server.address 127.0.0.1 --server.port 8501
pause
