# Assignment 4 - 9/7

To clone, make sure that you clone with `--recursive`.  If you forgot to do so, please execute the following commands
```
git submodule init
git submodule update
```

## Debugging
For this assignment, we have provided you a devcontainer that has additional debug tools installed.  To be able to use it, make sure you are connected to your Qualcomm Board using ssh with Visual Studio Code.  Then, ensure you have the `ms-vscode-remote.remote-containers` extension installed.

Once installed, you can click the blue remote connections button in the bottom-left corner of your Visual Studio Code window.  Then select "Reopen in Container".  This will leverage docker on your Qualcomm RB3 to start up a new container.

Once in the container, run `clinfo`.  Note that you now have two providers available to you.  You have the previous Qualcomm Adreno drivers, but now you also have pocl installed.  Note that Pocl has a platform index of 1 and a device index of 0.  You can force your code to use this driver by setting the `PLATFORM_INDEX` and `DEVICE_INDEX` environment variables.  Here is an example

```
PLATFORM_INDEX=1 DEVICE_INDEX=0 make run
```

I would leverage pocl for `printf` style debugging.  If you would like to leverage `valgrind` and `gdb`, these are also technically supported.

`oclgrind` is another tool that can be used for debugging.  To use it like you would `gdb`, execute `oclgrind -i ...` for interactive mode.

## Lab
There are two labs this class.  The first is based on the 2d convolution, see more information in the [README](lab/convolution-2d/README.rst).  The second implements a matrix multiply using `CLBlast`, see more information in the [README](lab/blas_mm/README.md).

## Homework
The homework will be building upon the 2d convolution you previously built.  See more information in the [README](homework/cnn/README.rst).