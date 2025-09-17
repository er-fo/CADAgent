@echo off
setlocal

set "FUSION_ADDINS_DIR=%APPDATA%\Autodesk\Autodesk Fusion 360\API\AddIns"
set "SOURCE_DIR=%~dp0"
set "TARGET_DIR=%FUSION_ADDINS_DIR%\CADAgent"

if not exist "%FUSION_ADDINS_DIR%" (
    mkdir "%FUSION_ADDINS_DIR%"
)

if exist "%TARGET_DIR%" (
    rmdir /s /q "%TARGET_DIR%"
)

xcopy "%SOURCE_DIR%*" "%TARGET_DIR%\" /e /i /q /y

endlocal
