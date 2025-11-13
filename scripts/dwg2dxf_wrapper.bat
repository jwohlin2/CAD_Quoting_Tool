@echo off
REM DWG to DXF conversion wrapper for ODA File Converter
REM Usage: dwg2dxf_wrapper.bat <input.dwg> <output.dxf>

setlocal enabledelayedexpansion

set "INPUT_DWG=%~1"
set "OUTPUT_DXF=%~2"
set "ODA_CONVERTER=D:\ODA\ODAFileConverter 26.8.0\ODAFileConverter.exe"

REM Check if input file exists
if not exist "%INPUT_DWG%" (
    echo ERROR: Input file not found: %INPUT_DWG% >&2
    exit /b 1
)

REM Get the input directory and filename
for %%F in ("%INPUT_DWG%") do (
    set "INPUT_DIR=%%~dpF"
    set "INPUT_NAME=%%~nxF"
)

REM Get the output directory
for %%F in ("%OUTPUT_DXF%") do (
    set "OUTPUT_DIR=%%~dpF"
)

REM Run ODA File Converter
REM Syntax: ODAFileConverter.exe <input_folder> <output_folder> <output_version> <output_format> <recurse> <audit> <input_filter>
"%ODA_CONVERTER%" "%INPUT_DIR%" "%OUTPUT_DIR%" "ACAD2018" "DXF" "0" "0" "%INPUT_NAME%"

if errorlevel 1 (
    echo ERROR: ODA File Converter failed >&2
    exit /b 1
)

echo SUCCESS: Converted %INPUT_DWG% to %OUTPUT_DXF%
exit /b 0
