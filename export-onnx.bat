@echo off
setlocal

:: Check if both parameters are provided
if "%~2"=="" (
    echo Usage: %0 ^<directory_name^> ^<config_file_name^>
    echo Example: %0 yolox_nano_poop_mixed_single yolox_custom_nano
    exit /b 1
)

:: Define the source path for the model's checkpoint file
set "SOURCE_PATH_BEST=YOLOX_outputs\%~1\best_ckpt.pth"
set "SOURCE_PATH_LAST=YOLOX_outputs\%~1\latest_ckpt.pth"
set "DEST_PATH_BEST=%~1_best"
set "DEST_PATH_LAST=%~1_last"

:: Copy the model checkpoint file
@REM echo Copying files
@REM copy "%SOURCE_PATH_BEST%" "%DEST_PATH_BEST%.pth"
@REM copy "%SOURCE_PATH_LAST%" "%DEST_PATH_LAST%.pth"

:: Run the Python command with the specified configuration file and output name
echo Running Python command to export to ONNX...
python tools/export_onnx.py --output-name "%DEST_PATH_BEST%.onnx" -f .\exps\default\%~2.py -c "%SOURCE_PATH_BEST%" -o 11
python tools/export_onnx.py --output-name "%DEST_PATH_LAST%.onnx" -f .\exps\default\%~2.py -c "%SOURCE_PATH_LAST%" -o 11

echo Operation completed.
endlocal
