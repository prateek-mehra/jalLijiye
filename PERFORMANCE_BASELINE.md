# Runtime baseline

Captured 2026-10-20 on an Apple M1 MacBook Air (macOS 26.3, Python 3.12.6).

## Manual-mode controller

| Metric | Measurement |
| --- | ---: |
| Average CPU over 30 seconds | 0.1% of one CPU core |
| Peak resident memory | 302 MiB |

This sample ran `JalLijiyeController` for 30 seconds with the camera/OpenCV detector unavailable (manual mode). CPU is the process user plus system CPU time divided by the 30-second wall-clock interval. Memory is the process peak resident set size reported by the operating system.

## Vision-enabled app status

After camera and logging access were granted in the user's Terminal, the documented `python -m app.main` process ran with camera inference active. Activity Monitor samples during the user's drinking-action attempt showed:

| Metric | Observed range |
| --- | ---: |
| CPU (% of one core) | 138–149% |
| Real memory | 214–218 MB |

These are point-in-time samples, not an interval average or peak. The process disappeared from Activity Monitor after the user exited it. Activity Monitor alone does not confirm whether the drinking event was detected.

## Vision-enabled app with debug window disabled

On 2026-10-20, a short comparison run with `show_debug_window: false` showed **131–136% CPU** across two Activity Monitor samples. Real memory fell from **306 MB to 244 MB** between samples, so this was still settling and is not a stable average or peak. This is an early comparison only; a longer run is needed to estimate steady-state memory and CPU. The config setting has since been restored to `true`.

The earlier attempt, before access was granted, failed because macOS denied camera access and blocked writing to `~/Library/Logs/JalLijiye/events.log`. The manual-mode controller sample above remains useful as a separate comparison. Its log directory was redirected to `/private/tmp/jallijiye-baseline/logs`; no project settings or tracked application files were changed for that sample.
