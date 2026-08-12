@echo off
REM ===================================================================
REM  OneVoice - chay thiet bi bang MOT lenh:  start.bat
REM
REM  Tu lo: venv, UTF-8 cho console Windows, tu nhan micro.
REM  Moi tham so them vao deu chuyen thang cho run_pipeline.py, vi du:
REM      start.bat --force-lang zh
REM      start.bat --show-route
REM ===================================================================
setlocal
cd /d "%~dp0"

REM Console Windows mac dinh la codepage 437/1258 -> tieng Viet va chu Han ra
REM dau hoi. Phai doi sang UTF-8 TRUOC khi Python ghi ra man hinh.
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

set PY=.venv\Scripts\python.exe
if not exist "%PY%" (
    echo [!] Chua co moi truong .venv
    echo     Chay lan dau:
    echo         python -m venv .venv
    echo         .venv\Scripts\python.exe -m pip install -r requirements.txt
    echo         .venv\Scripts\python.exe scripts\setup.py
    exit /b 1
)

if not exist "models\mt\nllb-zhvi-int8\model.bin" (
    echo [!] Chua co model. Dang tai tu dong...
    "%PY%" scripts\setup.py
    if errorlevel 1 exit /b 1
)

"%PY%" run_pipeline.py %*
