@echo off
title Unificador Corporativo PRO - Auto Launcher
echo ===============================================
echo     Iniciando Unificador Corporativo PRO...
echo ===============================================
echo.

REM Moverse a la carpeta donde está este .bat
cd /d "%~dp0"

REM 1. Verificar si Python está instalado
echo Verificando Python...
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python no está instalado en este equipo.
    echo Instala Python 3.10 desde: https://www.python.org/downloads/
    pause
    exit /b
)

REM 2. Crear entorno virtual si no existe
if not exist venv (
    echo Creando entorno virtual...
    python -m venv venv
)

REM 3. Activar entorno virtual
echo Activando entorno virtual...
call venv\Scripts\activate.bat

REM 4. Instalar dependencias si no están instaladas
echo Instalando dependencias...
pip install --upgrade pip
pip install -r requirements.txt

REM 5. Ejecutar la aplicación
echo.
echo Ejecutando la aplicación...
python -m streamlit run app.py

echo.
pause