# ZUNA 1.1 coordinate evidence — 2026-08-21

## Decision

Use MNE `standard_1005` head coordinates exactly as serialized to FIF, and
replicate the pinned ZUNA 1.1.3 tokenizer's componentwise discrete-bin clamp.
Store original coordinates, componentwise-clipped model coordinates, and final
integer XYZ tokens. Fail if two active channels collide on the same complete XYZ
token triplet.

This replaces the earlier blanket out-of-bounds rejection. It does not introduce
a geometric rescale or a target-informed transform.

## Primary evidence

1. The official ZUNA README's “Setting Montages” section instructs users to call
   `raw.set_montage(mne.channels.make_standard_montage("standard_1005"))` and
   says any montage with known positions is supported:
   <https://github.com/Zyphra/ZUNA#setting-montages>
2. The pinned `zuna==1.1.3` source function
   `eeg_data.py::discretize_chan_pos` normalizes against the configured XYZ
   extremes, converts to integer bins, warns when positions exceed those
   extremes, and explicitly calls `torch.clamp(..., 0, num_bins - 1)`.
3. The pinned inference config selects 100 bins and the `twelves` XYZ extremes,
   corresponding to -0.12 through +0.12 metres per axis.

## Current montage check

The verified Stage-0 tensor for `G001Day1Rest1.cnt` has 62 channels. Nine
channels exceed +0.12 m on Z and therefore receive Z token 99: FCz, C1, Cz, C2,
CP1, CPz, CP2, Pz, and P2. Their X/Y bins differ. All 62 complete XYZ token
triplets are unique; no active channel is spatially indistinguishable after the
official discretization.

## Required provenance and tests

- Cache identity includes original positions, clipped model positions, and
  discrete positions.
- Reconstruction manifests store all three representations and the transform
  name `official_zuna_componentwise_discrete_bin_clamp`.
- Any change to original positions changes cache identity even if it falls in
  the same saturated bin.
- Any complete XYZ token collision fails before model execution.
