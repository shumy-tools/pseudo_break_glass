# pseudo_break_glass
Pseudonym Identifier Derivation (P-ID) for Break-the-Glass

## Abstract
Pseudonymisation is a major requirement in recent data protection regulations, and of especial importance when sharing healthcare data outside of the boundaries of the affinity domain. However, healthcare systems require important break-the-glass procedures, such as accessing records of patients in unconscious states. Our work presents a pseudonymisation protocol that is compliant with break-the-glass procedures, established on a (t,n)-threshold secret sharing scheme and public key cryptography. The pseudonym is safely derived from a fragment of public information without any private secret requirement. The protocol is proven secure and scalable under reasonable assumptions.

## Dependencies
* rustc 1.36.0
* cargo 1.36.0

## Build
Build with release for optimal results.

```
cargo build --release
```

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
