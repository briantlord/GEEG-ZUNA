# Neuroscan CNT integer-width evidence — 2026-08-21

## Current decision

Keep the production gate blocked pending acquisition-system confirmation or a
verified export comparison. The available evidence strongly favors `int32` for
`G001Day1Rest1.cnt`, but the file's inconsistent event-table/footer layout
prevents MNE's independent `auto` inference from deciding on its own.

## External primary/implementation evidence

1. MNE's current Neuroscan CNT reader supports `auto`, `int16`, and `int32`.
   For files below 2 GB, `auto` derives bytes per sample from the header sample
   count, channel count, and event-table offset. The forced readers use little-
   endian signed two- or four-byte integers:
   <https://github.com/mne-tools/mne-python/blob/main/mne/io/cnt/_utils.py>
   and <https://github.com/mne-tools/mne-python/blob/main/mne/io/cnt/cnt.py>.
2. EEGLAB's Neuroscan loader independently exposes `int16` and `int32` and says
   to use `int32` for 32-bit data:
   <https://github.com/sccn/neuroscanio/blob/master/loadcnt.m>.
3. The Compumedics Neuroscan FAQ states that 32-bit CNT files are produced by
   SCAN 4.3.1 and newer:
   <https://compumedicsneuroscan.com/wp-content/uploads/3502D-Neuroscan-FAQs.pdf>.

These sources establish that `int32` is a real vendor format and that an
explicit width is sometimes necessary. They do not, by themselves, prove the
width of this specific acquisition.

## File-specific evidence for G001Day1Rest1.cnt

- File size: 179,459,525 bytes; header revision string: `Version 3.0`.
- Header: 67 channels, 667,099 samples, 1,000 Hz.
- MNE `data_format="auto"` fails because the header event-table position implies
  179,118,042 data bytes, which is not divisible by 67 channels. This is
  consistent with the independently observed abnormal footer/tail and means
  `auto` is not evidence for either width here.
- Forced `int32` and `int16` both use the same header duration (667.098 s), but
  event byte offsets map differently:

| Interpretation | annotations | event-1 count | median event-1 interval | last event-1 onset |
|---|---:|---:|---:|---:|
| int32 | 1,006 | 122 | 0.501 s | 104.871 s |
| int16 | 473 | 122 | 1.002 s | 209.743 s |

- In a 60-second analysis segment, the median EEG-channel SD is approximately
  29.9 microvolts under `int32` versus 264.9 microvolts under `int16`.

The exact twofold event timing and much less plausible `int16` scale strongly
favor `int32`. Final release still requires one of:

1. the acquisition/export setting showing 32-bit CNT;
2. a verified read/export from the acquisition software matching MNE `int32`;
3. a documented paradigm timing specification confirming the 0.501-second
   event spacing together with the file-specific header/byte analysis.

The same evidence check must be run across all 42 recordings; one recording
cannot establish cohort-wide encoding.
