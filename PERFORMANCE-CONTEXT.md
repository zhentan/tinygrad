# What the transfer numbers mean

The final qualification measured ten 2.00 MB transfers per direction in the existing USB timing tests. The warm statistic is the median of samples 2–10: **98.20 MB/s upload and 1.56 MB/s download**. Median measured times were 20.37 ms and 1,284.30 ms. These include the tinygrad host/device transfer path, staging, command submission and completion; they are not direct cable-bandwidth measurements.

USB-IF describes USB 10Gbps as a signaling rate. Dividing by eight gives a **nominal 1,250 MB/s upper bound before encoding/protocol overhead**. The recorded rates are 7.86% and 0.125% of that bound. This bound is useful context, not a promise that this bridge or software implementation can deliver it. Both rates are low relative to the link, and download is about 63 times slower than upload. [USB-IF terminology](https://www.usb.org/sites/default/files/usb_data_performance_language_usage_guidelines_jan_2024.pdf)

| Payload | Upload at recorded warm rate | Download at recorded warm rate |
| --- | ---: | ---: |
| 2.00 MB, measured | 20.37 ms | 1.284 s |
| 1 GB decimal, extrapolated | about 10.2 s | about 10.7 minutes |

The 1 GB row is an arithmetic extrapolation, not a benchmark result. Larger transfers and different workloads can behave differently. The cache correction recovered upload from 3.43 MB/s to 98.20 MB/s, but upload remains about 8.8% below the earlier 107.65 MB/s median. No matched repeated experiment has isolated that residual difference.

## Why download takes a different path

The native NV implementation uploads through vendor command F2: USB bulk OUT fills reserved controller SRAM and the GPU copy engine transfers the data to VRAM. Downloads first wait for the compute producer, copy to a VRAM readback window and observe completion, then use F0 mode 2 to stream PCIe reads into a host buffer. This branch uses 256 KiB staging windows. [NV staging implementation](https://github.com/zhentan/tinygrad/blob/8eb65a7d20a9de40a5ad2ebee42fd8ab2fddc25a/tinygrad/runtime/support/nv/usb.py#L122)

The pinned Chestnut firmware makes the distinction concrete. `do_usb_bulk_in` calls `pcie_read_chunk` to fill at most 1,024 bytes for SuperSpeed before arming each USB IN packet; the next packet is prepared after completion. `pcie_read_chunk` runs on the 8051 controller and reads four bytes through the PCIe transaction registers per loop, with address updates and completion polling. It overlaps the next PCIe request with storing the previous four bytes, but still handles payload in dwords rather than using the SRAM DMA path used for upload. [Pinned firmware packet handling](https://github.com/tinygrad/asm2464pd-firmware/blob/ed4e39b7e0794e19ba193477067c48757a5cf9ef/handmade/src/main.c#L135), [pinned programmed-I/O loop](https://github.com/tinygrad/asm2464pd-firmware/blob/ed4e39b7e0794e19ba193477067c48757a5cf9ef/handmade/src/pcie_pio.h#L27)

A 2,000,000-byte payload therefore requires 500,000 four-byte iterations in that firmware path, plus USB packet handling and host/GPU staging work. This is a strong architectural explanation for the asymmetry. **The exact fraction of elapsed time spent in firmware reads, completion waits, host compilation or USB calls has not been measured.** No cable fault, hard throughput ceiling or proven future speedup is inferred from these numbers. Removing the compute-producer wait is not a valid optimization: that wait fixed demonstrated incorrect readback.

## How this affects the PR's value

The evidence supports a functional initial RTX 3090 backend with substantial correctness coverage: 520 hardware tests, all nine USB checks, 64 MiB full-data integrity and one fresh-process check on the final head. The separate 100-run qualification belongs to the preserved pre-rebase head. These results are meaningful, but do not establish general hardware support, sustained multi-day reliability, application speedup or high transfer performance.

Keeping weights and intermediate tensors on the GPU, performing substantial computation there and returning a small result could amortize the transfer cost. Repeated large host readbacks, CPU/GPU offload and frequent materialization of tensors on the CPU are poor fits for the measured rates. These are workload implications, not measured application results.

Before advertising a performance benefit, the most useful additional evidence would be an end-to-end representative workload against the CPU baseline, reporting startup separately from steady-state execution and stating how much data crosses USB. To diagnose the existing download cost, a bounded timing breakdown of staging, waits and F0 bulk read is the next measurement. Neither additional experiment was run for this PR update.
