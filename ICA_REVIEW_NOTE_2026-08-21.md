# ICA artifact-cleaning review note — 2026-08-21

## Current verdict

Production remains blocked. The current automatic policy removes 14 of 20 ICA
components (70%) from `G001Day1Rest1.cnt`. This is too consequential to accept
without component-level review and a predeclared cohort rule.

## Frozen detector configuration

- MNE FastICA, 20 components, random state 0.
- Ocular detection: `find_bads_eog` using HEOG and VEOG, adaptive z-score
  measure, threshold 3.0, default 1–10 Hz scoring band.
- Muscle detection: `find_bads_muscle`, threshold 0.5, 7–45 Hz.

MNE documents the muscle score as the product of spectral slope, peripheral
focus, and spatial non-smoothness criteria when sensor positions exist. With
three criteria, the implementation compares the product against
`threshold ** 3`; at threshold 0.5, the effective product cutoff is 0.125:
<https://github.com/mne-tools/mne-python/blob/main/mne/preprocessing/ica.py>.

## One-recording result

- Ocular: components 0 and 2.
- Muscle: components 2, 7–17, and 19 (13 total).
- Union removed: components 0, 2, 7–17, and 19 (14 total).
- Component 0 has absolute VEOG correlation about 0.507 and a frontal
  topography, supporting its ocular label.
- Component 2 is labeled by both detectors, with maximum absolute ocular score
  about 0.116 and muscle product score about 0.234.
- Muscle product scores range from about 0.234 to 0.840 among excluded muscle
  components. Because the implemented cutoff is 0.125, several moderate-score
  components are removed as designed; the list is not a software threshold bug.

The generated review table is
`results/ica_review_G001Day1Rest1.csv` locally. It includes component index,
detector labels/scores, five strongest topography channels, PCA explained
variance, and blank reviewer decision/notes fields.

## Required decision before release

An independent reviewer should mark each component retain/remove/uncertain using
topography, source time series, spectrum, and detector scores. Then freeze one
cohort-wide policy before inspecting downstream ZUNA preservation outcomes. At
minimum, compare:

1. ocular-only removal;
2. ocular plus reviewer-confirmed muscle removal;
3. the current automatic ocular-plus-muscle rule as a sensitivity.

Do not choose the policy based on which one makes ZUNA look best. If the
automatic rule remains primary, add a predeclared maximum exclusion-fraction
gate or a manual-review trigger; 70% removal must never pass silently.
