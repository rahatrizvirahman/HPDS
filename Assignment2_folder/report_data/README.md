# Report data

Copies of the audited CSV files from the Athena evidence, for building report tables and figures. Every row is one individual
measurement (raw) or a recomputed summary. The full evidence (logs, predictions) is in the archives named in `REPORT_HANDOFF.md`.
Hardware differs between files and must stay labeled. Do not pool files. Times are milliseconds.

| File | Source | SHA-256 |
|---|---|---|
| `athena_a100_global_raw.csv` | A100, job 3382505, global kernel, blocks 64/128/256, athena521; original `HPDS_assignment2/results/athena/baseline-20261022T220759Z-job3382505-biaftwm5/raw.csv` | `e2522d3a1e048ba0106c5f9365a07403968f9b123f51fdfd516a1329149cebc3` |
| `athena_a100_global_summary.csv` | A100, job 3382505, global kernel, blocks 64/128/256, athena521; original `HPDS_assignment2/results/athena/baseline-20261022T220759Z-job3382505-biaftwm5/summary.csv` | `c181f7273d5d223df2e8fb961b30d874cd9017528300c9f6105574c40742cf84` |
| `athena_a100_global_kernel-summary.csv` | A100, job 3382505, global kernel, blocks 64/128/256, athena521; original `HPDS_assignment2/results/athena/baseline-20261022T220759Z-job3382505-biaftwm5/kernel-summary.csv` | `338b24af45d67b014040ec76bb5c4dbb106e1ad3da3e2adc1f262bdbee98ba9e` |
| `athena_a100_global_shared_raw.csv` | A100, job 3382591, global and shared kernels, athena521; original `HPDS_assignment2/results/athena-shared/baseline-20261022T233959Z-job3382591-0hypd1j1/raw.csv` | `db6b0c641820956cad6b669ed70cf92bfd90ffc3cac7f12f79b402b4740b84d0` |
| `athena_a100_global_shared_summary.csv` | A100, job 3382591, global and shared kernels, athena521; original `HPDS_assignment2/results/athena-shared/baseline-20261022T233959Z-job3382591-0hypd1j1/summary.csv` | `9a2ab954b2221a281f26ee9390e99a574d38f119b66d66a7d9fc9c24ae6f0160` |
| `athena_a100_global_shared_kernel-summary.csv` | A100, job 3382591, global and shared kernels, athena521; original `HPDS_assignment2/results/athena-shared/baseline-20261022T233959Z-job3382591-0hypd1j1/kernel-summary.csv` | `d43104501eac7d6a7189118774b24fe5868cbb6769ca7dd81899cc65d9248a21` |
| `athena_h100_global_shared_raw.csv` | H100, job 3382996, global and shared kernels, athena533; original `HPDS_assignment2/results/athena-shared-h100/baseline-20261023T142149Z-job3382996-mw1k4vro/raw.csv` | `60e47e6c64994047f2eaf54cc67cc1f03688c26a830a5ce58d2ab0984cd00523` |
| `athena_h100_global_shared_summary.csv` | H100, job 3382996, global and shared kernels, athena533; original `HPDS_assignment2/results/athena-shared-h100/baseline-20261023T142149Z-job3382996-mw1k4vro/summary.csv` | `54aaaafb9ce86e0a938a77df8a64c5952e6318389a5a6bee9493dc055673f95a` |
| `athena_h100_global_shared_kernel-summary.csv` | H100, job 3382996, global and shared kernels, athena533; original `HPDS_assignment2/results/athena-shared-h100/baseline-20261023T142149Z-job3382996-mw1k4vro/kernel-summary.csv` | `53ced8dfe1ba9c987b14431135f8a25c6b45e4780beebcc0b226df50330ebecf` |
| `athena_cpu_raw.csv` | CPU, job 3383699, serial/pthreads/OpenMP/MPI, athena201 (AMD EPYC 7742, 16 CPUs); original `HPDS_assignment2/results/athena-cpu/cpu-benchmark-20261023T192218Z-job3383699-i4cdjm1w/raw.csv` | `9e3b0f7f1917fb2e19a57cd5e1ed486e23d331a9b1fe8e3de288317256b23db5` |
| `athena_cpu_summary.csv` | CPU, job 3383699, serial/pthreads/OpenMP/MPI, athena201 (AMD EPYC 7742, 16 CPUs); original `HPDS_assignment2/results/athena-cpu/cpu-benchmark-20261023T192218Z-job3383699-i4cdjm1w/summary.csv` | `1085d0a6f48687e1a56ad8450fcfb15ced5b3ec7ba076959a3ab06fc3997e899` |
