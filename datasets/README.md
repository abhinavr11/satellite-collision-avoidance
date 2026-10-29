# Versioned demonstration data

[`swarm_ac_shifted_v1`](swarm_ac_shifted_v1/README.md) is the single input case
used by the Colab notebook and the encounter trajectory/probability code. It
contains Swarm A and C precise reference and SGP4 trajectories on one synthetic
timeline, the original native timestamps, the two selected TLEs, two original
daily SP3 files, the pinned Earth-orientation input and source hashes.

The two-hour case is intentionally included in Git so cloning this repository
is sufficient to run both notebook deliverables without downloading the larger
source collection. It is a constructed time-shift encounter, not a recorded
historical collision. Reference states are processed GPS-based precise orbits,
not raw receiver observations. No artificial spatial noise is added.

CSV input bytes are preserved by `.gitattributes` because their SHA-256 hashes
are recorded in `encounter.json`. Keep this folder together when copying it.
Source-data provider terms and notices remain applicable; see
[`docs/DATA_SOURCES.md`](../docs/DATA_SOURCES.md).

The full historical collections and environments remain excluded from Git.
