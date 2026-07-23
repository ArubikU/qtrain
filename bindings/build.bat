@echo off
REM Build qubit_native.pyd directly with cl inside the VS2019 build env.
REM setuptools' VS auto-detection misses the 2019 BuildTools install, so we
REM invoke the compiler ourselves. Run from the qtrain/ directory.

call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (echo vcvars failed & exit /b 1)

set PYINC=C:\Users\ejane\AppData\Local\Programs\Python\Python312\Include
set PYLIB=C:\Users\ejane\AppData\Local\Programs\Python\Python312\libs
set PB11=C:\Users\ejane\AppData\Local\Programs\Python\Python312\Lib\site-packages\pybind11\include

cl /nologo /LD /EHsc /std:c++17 /O2 /DNDEBUG /openmp ^
   /I"%PYINC%" /I"%PB11%" /I"..\include" ^
   bindings\qubit_py.cpp ^
   /Fe:qubit_native.pyd ^
   /link /LIBPATH:"%PYLIB%"
exit /b %errorlevel%
