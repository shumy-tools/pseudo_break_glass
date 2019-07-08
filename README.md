# pseudo_break_glass
Pseudonym Identifier Derivation (P-ID) for Break-the-Glass

## Dependencies
rustc 1.35.0
rustc 1.35.0

## Usage
This project is a tool to measure running times of the proposed P-ID scheme. The tool accepts parameters to setup the number of parties (n) and the threshold value (t).

```
Statistics for P-ID 1.0
Micael Pedrosa <micaelpedrosa@ua.pt>
Performs time measurements for runs of the PID function.

USAGE:
    pseudo-id --parties <parties> --threshold <threshold>

FLAGS:
    -h, --help       Prints help information
    -V, --version    Prints version information

OPTIONS:
    -p, --parties <parties>        Sets the number of parties (n)
    -t, --threshold <threshold>    Sets the threshold number (t)
```

with the result output:

```
Setup: (parties=<n>, threshold=<t>)
Avg. per run (500 runs): <x>ms
```
