@echo off
REM build_kernel.bat - build the minimal FalcoCV native kernel (_falco_kernel)
REM
REM Usage:
REM   build_kernel.bat            configure + build + deploy
REM   build_kernel.bat --rebuild  clean build cache first
REM   build_kernel.bat --clean    remove build artifacts

setlocal enabledelayedexpansion
cd /d "%~dp0"

set "BUILD_DIR=%CD%\build\kernel"
set "BIN_DIR=%CD%\cpp\bin"
set "CMAKE_GEN=Visual Studio 17 2022"
set "CMAKE_ARCH=x64"

if /I "%1"=="--help" (
    echo Usage: build_kernel.bat [--rebuild --clean]
    exit /b 0
)

if /I "%1"=="--clean" (
    if exist "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%"
    if exist "%BIN_DIR%\_falco_kernel.pyd" del /q "%BIN_DIR%\_falco_kernel.pyd"
    echo [CLEAN] Done.
    exit /b 0
)

if /I "%1"=="--rebuild" (
    if exist "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%"
)

echo [CHECK] pybind11...
py -3.13 -c "import pybind11" 2>nul
if errorlevel 1 (
    echo [INSTALL] pybind11 not found...
    py -3.13 -m pip install pybind11
    if errorlevel 1 exit /b 1
)

echo [CHECK] CMake...
where cmake >nul 2>&1
if errorlevel 1 (
    echo [FATAL] CMake not found.
    exit /b 1
)

for /f "delims=" %%p in ('py -3.13 -c "import sys; print(sys.executable)"') do set "PYTHON313=%%p"
echo [OK] Python at %PYTHON313%

echo [CMAKE] Configuring...
cmake -S cpp/kernel -B "%BUILD_DIR%" -G "%CMAKE_GEN%" -A "%CMAKE_ARCH%" ^
    -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="%PYTHON313%"
if errorlevel 1 exit /b 1

echo [CMAKE] Building...
cmake --build "%BUILD_DIR%" --config Release
if errorlevel 1 exit /b 1

if not exist "%BIN_DIR%" mkdir "%BIN_DIR%"
for %%f in ("%BIN_DIR%\Release\_falco_kernel.*.pyd" "%BIN_DIR%\_falco_kernel.*.pyd" "%BUILD_DIR%\Release\_falco_kernel.*.pyd") do (
    if exist "%%f" (
        copy /y "%%f" "%BIN_DIR%\_falco_kernel.pyd" >nul
        if /I not "%%~dpnxf"=="%BIN_DIR%\_falco_kernel.pyd" del "%%f" >nul 2>&1
        echo [OK] _falco_kernel.pyd deployed
        goto :done
    )
)
echo [WARN] _falco_kernel.pyd not found
exit /b 1

:done
echo [DONE] %BIN_DIR%\_falco_kernel.pyd
