@echo off
if not exist bin mkdir bin
pushd bin

@REM del *.exe *.dll *.pdb *.obj *.ilk

@REM cl /nologo /O2 ../src/train_1.c

@REM cl /nologo /O2 ../src/train_2.c

@REM cl /nologo /O2 ../src/handwritten_digit.c

@REM cl /nologo /O2 ../src/handwritten_digit_2.c

@REM cl /nologo /O2 ../src/handwritten_digit_batch.c

cl /nologo /O2 ../src/handwritten_digit_test.c

@REM cl /nologo /O2 /LD /Fe:libnn.dll ../src/nn.c

@REM if exist train_1.exe train_1.exe
@REM if exist train_2.exe train_2.exe
@REM if exist handwritten_digit.exe handwritten_digit.exe
@REM if exist handwritten_digit_2.exe handwritten_digit_2.exe
@REM if exist handwritten_digit_batch.exe handwritten_digit_batch.exe
if exist handwritten_digit_test.exe handwritten_digit_test.exe

popd bin
