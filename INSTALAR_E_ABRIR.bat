@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo.
echo ============================================
echo  mtdrop - instalacao e interface local
echo  Build: 2026-10-09-batch-ui-v3
echo ============================================
echo.
echo Pasta: %CD%
echo Se esta pasta for antiga, baixe de novo mtdrop-windows.zip
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
echo A UI escolhe uma porta livre automaticamente (7860-7870).
echo Leia nesta janela a linha:
echo.
echo   mtdrop UI -^> http://127.0.0.1:XXXX
echo.
echo e abra ESSE URL no navegador (pode nao ser 7860).
echo.
echo Se a porta estiver ocupada por um mtdrop antigo:
echo   - Feche a outra janela preta (Ctrl+C) e tente de novo, ou
echo   - Deixe este script; ele usa a proxima porta livre.
echo.
echo Deixe esta janela aberta enquanto usar o mtdrop.
echo Feche com Ctrl+C para encerrar.
echo.

REM Prefer 7860; Python falls back through 7870 if busy.
REM Override manually, e.g.: set MTDROP_PORT=7875
if defined MTDROP_PORT (
  set "PORT_ARGS=--port %MTDROP_PORT%"
) else (
  set "PORT_ARGS=--port 7860 --port-span 11"
)

%PY% -m mtdrop ui --host 127.0.0.1 %PORT_ARGS%
if errorlevel 1 (
  echo.
  echo [ERRO] Nao foi possivel iniciar a UI.
  echo Confirme a instalacao. Se a porta falhar, feche outras janelas
  echo do mtdrop ou rode:  %PY% -m mtdrop ui --port 0
  echo.
  pause
  exit /b 1
)

endlocal
