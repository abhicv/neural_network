@echo off
if not exist bin mkdir bin
pushd bin

@REM del *.exe *.dll *.pdb *.obj *.ilk

@REM cl /nologo /O2 ../src/regression_demo.c

@REM cl /nologo /O2 ../src/simple_regression_demo.c

cl /nologo /O2 ../src/handwritten_digit_train.c

@REM cl /nologo /O2 ../src/handwritten_digit_test.c

@REM cl /nologo /O2 /LD /Fe:libnn.dll ../src/nn.c

@REM if exist regression_demo.exe regression_demo.exe
@REM if exist simple_regression_demo.exe simple_regression_demo.exe
if exist handwritten_digit_train.exe handwritten_digit_train.exe
@REM if exist handwritten_digit_test.exe handwritten_digit_test.exe

popd bin
