# Tests with Catch
The tests are written by utilizing the Catch framework, a header-only test
framework. The main test file includes all test header files to be able to use
the command line tool that is added with the ```#define CATCH_CONFIG_MAIN```.
See the Catch framework on github for more information about this.

## How to run
Until a script is added to run these test aumatically upon make, the tests can
be executed by:

~~~Shell
$ <eos_root_install_dir>/tests/main-tests
~~~

Note that we execute all tests with this command. For help, pass ``-h``` to the
main-test binary file.

~~~Shell
$ <eos_root_install_dir>/tests/main-tests -h
~~~

For example if the root install dir is right in this folder (as the main README
suggests with ```-DCMAKE_INSTALL_PREFIX```), then do:

~~~Shell
$ ./install/tests/main-tests'
~~~

If you are using the docker files from the docker branch, the command looks
like this:

~~~Shell
$ CMD='cd build/tests; make -j4 && make install && cd ../../ && ./install/tests/main-tests' make run-bash-cmd
~~~

## References
* Catch on github [Catch framework](https://github.com/philsquared/Catch)
