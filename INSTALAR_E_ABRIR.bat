@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo.
echo ============================================
echo  mtdrop - instalacao e interface local
echo ============================================
echo.
echo Pasta: %CD%
echo.

where py >nul 2>&1
if %ERRORLEVEL%==0 (
  set "PY=py -3"
) else (
  where python >nul 2>&1
  if %ERRORLEVEL%==0 (
    set "PY=python"
  ) else (
    echo [ERRO] Python nao encontrado.
    echo Instale Python 3.10+ de https://www.python.org/downloads/
    echo Marque "Add python.exe to PATH" na instalacao.
    echo.
    pause
    exit /b 1
  )
)

echo Usando: %PY%
echo.
echo [1/2] Instalando dependencias (pode demorar alguns minutos)...
%PY% -m pip install -e ".[ui]"
if errorlevel 1 (
  echo.
  echo [ERRO] Falha na instalacao com pip.
  echo Tente abrir um terminal nesta pasta e rode:
  echo   %PY% -m pip install -e ".[ui]"
  echo.
  pause
  exit /b 1
)

echo.
echo [2/2] Abrindo a interface...
echo.
echo Quando aparecer "Running on local URL", abra no navegador:
echo.
echo   http://127.0.0.1:7860
echo.
echo Deixe esta janela aberta enquanto usar o mtdrop.
echo Feche com Ctrl+C para encerrar.
echo.

%PY% -m mtdrop ui --host 127.0.0.1 --port 7860
if errorlevel 1 (
  echo.
  echo [ERRO] Nao foi possivel iniciar a UI.
  echo Confirme a instalacao e tente de novo.
  echo.
  pause
  exit /b 1
)

endlocal
