gcc -O3 -c -fpic nn.c -o nn.o
gcc -shared -o libnn.so nn.o