# DDXPlus Semantic Mapper Validation Gates

- primary model: `gpt-5.6-sol`
- auditor model: `gpt-5.4`
- all gates passed: **True**
- locked test read: **no**

| gate | metric | result | threshold | pass |
|---|---|---:|---:|---:|
| G1 | reader finding micro F1 | 1.0000 | >= 0.98 | True |
| G1 | reader native-value accuracy | 1.0000 | >= 0.98 | True |
| G2 | absent-target false map | 0.0000 | <= 0.05 | True |
| G3 | cache replay byte-identical | True | true | True |
| G3 | cold duplicate agreement | 1.0000 | report only | - |
| G4 | evidence disagreement | 0.0200 | <= 0.05 | True |
| G4 | conditional value disagreement | 0.0000 | <= 0.05; n >= 20 | True |
| G4 | primary value-reference accuracy | 1.0000 | >= 0.98 | True |
| G4 | auditor value-reference accuracy | 1.0000 | >= 0.98 | True |
