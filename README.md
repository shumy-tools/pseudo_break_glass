# Title
A pseudonymisation protocol with implicit and explicit consent routes for health records in federated ledgers

## Abstract
Healthcare data for primary use (diagnosis) may be encrypted for confidentiality purposes; however, secondary uses such as feeding machine learning algorithms requires open access. Full anonymity has no traceable identifiers to report diagnosis results. Moreover, implicit and explicit consent routes are of practical importance under recent data protection regulations (GDPR), translating directly into break-the-glass requirements. Pseudonymisation is an acceptable compromise when dealing with such orthogonal requirements and is an advisable measure to protect data. Our work presents a pseudonymisation protocol that is compliant with implicit and explicit consent routes. The protocol is constructed on a (t,n)-threshold secret sharing scheme and public key cryptography. The pseudonym is safely derived from a fragment of public information without requiring any data-subject's secret. The method is proven secure under reasonable cryptographic assumptions and scalable from the experimental results.

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
    pseudo-id --select <select> --threshold <threshold>

FLAGS:
    -h, --help       Prints help information
    -V, --version    Prints version information

OPTIONS:
    -s, --select <select>          Selects the simulation process. (k) - master key setup, (m) - multiparty computation
    -t, --threshold <threshold>    Sets the threshold number (t). The number of parties are set automatically to 3t+1.
```

with output examples:

```
Setup: (t=32, 3t+1=97)
---Multiparty computation stats (100 runs)---
   Avg. per run: (a_verif=0ms, s_verif=243ms, total=254ms)
```

```
Setup: (t=32, 3t+1=97)
---Master key setup stats---
   Matrix setup (size=294.03125Kb, time=1124ms)
   Commit and encrypt: 232ms
   Share verification: 16602ms
   Total: 17960ms
```
