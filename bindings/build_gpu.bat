@echo off
REM Build qubit_gpu_native.pyd with nvcc (CUDA 12.6, sm_86) inside the
REM VS2019 build env. Run from the qtrain/ directory:  cmd /c bindings\build_gpu.bat

call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (echo vcvars failed & exit /b 1)

set NVCC="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe"
set PYINC=C:\Users\ejane\AppData\Local\Programs\Python\Python312\Include
set PYLIB=C:\Users\ejane\AppData\Local\Programs\Python\Python312\libs
set PB11=C:\Users\ejane\AppData\Local\Programs\Python\Python312\Lib\site-packages\pybind11\include

%NVCC% -O2 -std=c++17 -arch=sm_86 --shared -DQUBIT_CUDA ^
   -Xcompiler "/openmp /EHsc /MD /utf-8 /DNDEBUG" ^
   -I"%PYINC%" -I"%PB11%" -I"..\include" ^
   bindings\qubit_gpu.cu ..\src\backend_gpu.cu ^
   -o qubit_gpu_native.pyd ^
   "%PYLIB%\python312.lib"
exit /b %errorlevel%
