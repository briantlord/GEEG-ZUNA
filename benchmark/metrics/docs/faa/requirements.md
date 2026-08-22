# FAA — requirements (reference metric)

*FAA is the reference plug-in used to validate the modular framework; it reproduces the original
`biomarker_eval.py` numbers exactly. This doc is written retrospectively to complete the 5-part record.*

1. **Motivation.** Frontal alpha asymmetry indexes approach/withdrawal motivation and affective style;
   relatively greater left-frontal alpha (lower left activity) is associated with withdrawal/depression
   risk. It is the canonical, easy-to-test psychophysiological metric and the anchor of this project.
2. **Definition.** `ln(alpha power, right) − ln(alpha power, left)`, 8–13 Hz, at mid-frontal **F3/F4**
   (`faa`, primary) and lateral-frontal **F7/F8** (`faa_lat`, secondary).
3. **Electrodes / drop set.** Computed from F3, F4, F7, F8. The preservation test drops exactly those
   four channels (`drop_channels = ['F3','F4','F7','F8']`) and reconstructs them.
4. **Submetrics & units.** `faa`, `faa_lat` — dimensionless log-power ratios (natural log).
5. **Reference frame.** **Current source density (surface Laplacian)** — reference-free, as Allen's group
   uses, because scalp FAA is otherwise strongly reference-dependent. CSD also makes FAA independent of the
   surviving-channel average-reference choice.
6. **Reliability / floor.** Same-day (Rest1↔Rest2) floor across 5 subjects: `faa` ≈ 0.208, `faa_lat` ≈ 0.301.
7. **Edge cases.** Skip a pair if either channel is absent; `+1e-20` log floor; return finite floats.
8. **References.** Coan & Allen 2004; Smith, Reznik, Stewart & Allen 2017.
