# pseudo_break_glass
Pseudonym Identifier Derivation (P-ID) function with break-the-glass compatibility.

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
This project is a tool to perform simulations and measurements for the FedPI protocols. The goal is to measure the scalability of the proposed protocols. The measurements do not take into account network latency.

```
Statistics for FedPI 1.0
Micael Pedrosa <micaelpedrosa@ua.pt>
Simulations and measurements for the FedPI protocol.

USAGE:
    pseudo-id --threshold <threshold>

FLAGS:
    -h, --help       Prints help information
    -V, --version    Prints version information

OPTIONS:
    -t, --threshold <threshold>    Sets the threshold number (t). The number of parties are set automatically to 3t+1.
```

with the result output example:

```
Setup: (t=32, 3t+1=97)
---Master key setup stats---
   Matrix setup (size=294.03125Kb, time=1181.195ms)
   Commit and encrypt: 247.326ms
   Share verification: 17810.898ms
   Total: 19244.28ms
---Multiparty computation stats (500 runs)---
   Avg. per run: 7.698006ms
```
