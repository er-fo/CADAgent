@echo off
setlocal

set "FUSION_ADDINS_DIR=%APPDATA%\Autodesk\Autodesk Fusion 360\API\AddIns"
set "SOURCE_DIR=%~dp0"
set "TARGET_DIR=%FUSION_ADDINS_DIR%\CADAgent"

REM Get the current folder name (might be CADAgent-main, CADAgent-1.0.0, etc.)
for %%I in ("%SOURCE_DIR%.") do set "CURRENT_FOLDER_NAME=%%~nxI"

if not exist "%FUSION_ADDINS_DIR%" (
    mkdir "%FUSION_ADDINS_DIR%"
)

if exist "%TARGET_DIR%" (
    rmdir /s /q "%TARGET_DIR%"
)

REM Copy the contents and ensure the target is named exactly "CADAgent"
xcopy "%SOURCE_DIR%*" "%TARGET_DIR%\" /e /i /q /y

REM If we're not already in a folder named "CADAgent", rename the parent folder too
if not "%CURRENT_FOLDER_NAME%"=="CADAgent" (
    for %%I in ("%SOURCE_DIR%..") do set "PARENT_DIR=%%~fI"
    set "NEW_SOURCE_DIR=%PARENT_DIR%\CADAgent"
    if not exist "%NEW_SOURCE_DIR%" (
        ren "%SOURCE_DIR%." "CADAgent" 2>nul
    )
)

endlocal
