# Data sources, downloads, and third-party notices

The selected shifted Swarm A/C input case is committed in [`datasets/swarm_ac_shifted_v1`](../datasets/swarm_ac_shifted_v1/README.md), including the original two daily SP3 files, two selected TLEs and pinned Earth-orientation table needed to rebuild it. Its `encounter.json` records source hashes and original collection paths. The larger source collections remain local. `config/source_manifest.json` contains source URLs, archived file sizes, hashes, download dates, and source categories from the October 8, 2026 collection. The Python downloader uses these recorded file URLs instead of a fragile archive-directory page number. Bulk downloads go to ignored local `data/` by default.

## Required experimental inputs

| Source | Product | Treatment |
|---|---|---|
| [ESA Swarm dissemination archive](https://swarm-diss.eo.esa.int/) | 102 daily reduced-dynamic SP3 orbit ZIPs for Swarm A/B/C, baseline 0203 | Fixed historical file URLs; archived SHA-256 required |
| [Research authors' TLE dataset](https://github.com/Jeong-Seung-Won/-IEEE-Access-Toward-Data-Driven-Satellite-Orbit-Prediction) | Swarm A/B/C historical TLEs | Pinned commit and archived SHA-256; each TLE checksum and Git blob identity checked during preparation |
| Same pinned repository | Git tree metadata | Used to check TLE Git blob identities |
| [IERS Earth-orientation table](https://datacenter.iers.org/data/9/finals2000A.all) | `finals2000A.all` | Upstream file can update; downloaded bytes and hash recorded |
| [IERS leap seconds](https://hpiers.obspm.fr/iers/bul/bulc/Leap_Second.dat) | Leap-second table | Upstream file can update; downloaded bytes and hash recorded |

The TLE source commit is `b1473e2cddbe660f0fde66c3abb8315beadb7653`. The authors identify Space-Track as their original source. This project uses their public research redistribution; it is not a fresh authenticated download directly from Space-Track. The original repository's license and README are preserved in `docs/source-notices/`. Do not treat that license as a blanket license for unrelated provider data.

The reference product is attributed to the European Space Agency and Delft University of Technology: [SW_SP3xCOM_2_, DOI 10.57780/esa-8579355](https://doi.org/10.57780/esa-8579355). Its [product description](https://swarm-disc.github.io/product-catalogue-tools/SW_SP3xCOM_2_.html) explains its precise orbit-determination inputs and units. The [archive](https://swarm-diss.eo.esa.int/) links ESA's data policy and terms. Keep provider attribution when reporting results.

## Optional full collection: `--all`

These inputs are organized into the same additional folders as the original collection, but are not used by the three numerical pilots:

- **ESA Collision Avoidance Challenge:** [Zenodo record 4463683](https://zenodo.org/records/4463683), DOI [10.5281/zenodo.4463683](https://doi.org/10.5281/zenodo.4463683), CC BY 4.0. Authors: Thomas Uriot, Dario Izzo, Jose Martinez-Heras, Francesca Letizia, Jan Siminski, and Klaus Merz. The full archive includes training data, test data, released private test labels, and the compressed original records. The archived SHA-256 is required. A field named `true_risk` is a final estimated risk, not evidence of an observed collision.
- **CelesTrak:** [SOCRATES](https://celestrak.org/SOCRATES/) screening records and [current GP elements](https://celestrak.org/NORAD/elements/). URLs return live snapshots, so a later download is not the same historical snapshot. Format and usage-policy pages are included in the source list. The repository does not assign these data a new license.
- **GFZ:** [Kp, ap, Ap, solar, and sunspot indices](https://kp.gfz.de/en/data), DOI [10.5880/Kp.0001](https://doi.org/10.5880/Kp.0001). The distributed file's header identifies CC BY 4.0, except sunspot numbers, which use CC BY-NC 4.0. The format description is preserved in `docs/source-notices/`.
- Provider documentation, source listings, and dataset metadata are fetched alongside these data.

The archive and extracted copies are intentionally both retained locally. The original full collection occupied about 946 MB. Current optional live snapshots can vary in size, rows, dates, and the selected current conjunctions.

## What can change on redownload

Historical TLE, precise-orbit, and ESA archive inputs must match archived hashes. A changed historical product is rejected, rather than silently substituted. Mutable Earth-orientation, live snapshot, and documentation endpoints can legitimately change. For them, the downloader records the actual new hash, timestamp, archived hash, and whether the bytes match the old snapshot.

The saved experimental results describe the earlier input collection. Fresh IERS tables can contain revisions; a later live CelesTrak snapshot cannot recreate the original 2026 screening list. Redownloading the same provider URLs therefore does not guarantee byte-for-byte reproduction of every file. Numerical regression checks allow small conversion differences but flag substantial result changes.

No account or payment step is performed by the downloader. Provider outages and format changes are reported as failures. Rerunning resumes from completed matching files. The manifests are metadata; they are not a substitute for downloaded data bytes.

## Original code

No new open-source license has been selected for the original course-project code. Third-party licenses and source terms remain applicable to their own material. Team context and links in the README are attribution, not a blanket redistribution license.
