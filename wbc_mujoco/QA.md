# Local QA record

Checked on 2026-08-23 with Google Chrome `151.0.7922.173`
headless/SwiftShader.

## Static, provenance, and integrity checks

- `npm run check`: pass (13 JavaScript contract tests, native controller/model-asset
  parity, and production Vite build)
- `npm audit --omit=dev`: 0 vulnerabilities
- interaction timing is separated by source: keyboard `0.00 s`, sliders `0.20 s`,
  validated presets `2.00 s` after their one-second neutral settle
- every preset command is inside the play limits, lies exactly on its slider grid,
  and only `turn_left` and `spin_backbend` request nonzero turn rate
- checkpoint training ranges and curriculum starts match the portable v7 range
  reference; this reference intentionally does not describe the v9 command-mode mix
- the two MJCF XMLs and all 36 referenced meshes match the native `sim2mujoco`
  copies byte-for-byte
- browser policy: `rhe0p5_ft50k_ci20-60_wrp150-4_hpr150-3_fcs0p5_jtl100_e12288_s1_v9`
  teacher at iteration `240000`; checkpoint SHA-256
  `9a003c7022df8279b2ca92e93caf60f065626f3aa78b10ca2a1a725f5753aa36`
  and ONNX SHA-256
  `2f02e63dcf77c39556a0cadd99e5a2a1c3062697e51911b278ce9ffb763c70d6`
  match the declared contract provenance
- the ONNX is opset 11 with fixed `obs [1,60] -> actions [1,15]`; all eight actor
  initializers are bit-exact with the checkpoint
- the browser fetches the ONNX bytes, verifies their full SHA-256, and only then
  creates the ONNX Runtime session; a deliberately tampered policy was rejected
- release `v9-rhe0p5-240000-catalog3` is pinned through the outer page, demo entry
  point, contract request, runtime probe, and policy-hash request; a deliberately
  stale warm-cache contract was rejected before any policy fetch rather than being
  mixed with the new release
- the webpage policy is intentionally independent from the sibling WBC repository's
  bundled v7 teacher, student, and native-policy example
- pinned stack: `mujoco-js@0.0.7`, `onnxruntime-web@1.21.1`,
  `three@0.181.0`, `vite@6.4.3`

The v9 moving-command population uses 40% standing environments. Of the remaining
60%, 30% use heading feedback, 15% use a direct exact-zero turn rate, and 15% use a
direct uniformly sampled turn rate.

## Runtime parity probe

- browser MuJoCo runtime: 3.3.8
- native Python MuJoCo runtime: 3.3.7
- initial 60-value observation: exact match
- initial 15-value policy action from the same webpage ONNX under CPU/WASM:
  maximum absolute difference `9.536743e-7` (mean `2.389153e-7`)
- production build loaded the versioned contract and exact policy, rendered the
  robot, initialized ONNX Runtime, and completed every query-driven QA replay with
  no application console, runtime, or browser-log errors; Chrome emitted only its
  non-fatal SwiftShader deprecation warning

The one-patch MuJoCo engine difference is intentional and documented; it is why the
browser stability screen below is separate from native validation.

## Browser stability and visual screen

Every validated preset ran 400 policy ticks / 8 simulated seconds using the one-second
neutral settle and two-second smootherstep ramp. All runtime commands exactly matched
the contract after Float32 conversion, and none triggered the fall detector.

| Preset | Min pelvis z | Final pelvis z | Final contact L/R | Fell |
|---|---:|---:|:---:|:---:|
| forward | 0.643547 | 0.677556 | 0 / 1 | no |
| backward | 0.643547 | 0.674012 | 1 / 0 | no |
| strafe_left | 0.643547 | 0.682722 | 0 / 1 | no |
| turn_left | 0.643547 | 0.668882 | 1 / 1 | no |
| squat_bow_walk | 0.321319 | 0.325507 | 1 / 1 | no |
| twisted_walk | 0.571467 | 0.574788 | 1 / 1 | no |
| side_lean_walk | 0.443960 | 0.450199 | 1 / 0 | no |
| pitch_strafe | 0.333933 | 0.336876 | 1 / 1 | no |
| forward_backbend | 0.540482 | 0.622077 | 0 / 1 | no |
| backlook_reverse | 0.573956 | 0.581863 | 0 / 1 | no |
| spin_backbend | 0.616811 | 0.623950 | 0 / 1 | no |
| tall_extension | 0.643547 | 0.662877 | 0 / 1 | no |

Desktop `1440x1000` and mobile `390x844` captures were inspected for all motions.
The robot remained visible and coherent, controls stayed legible, and no desktop or
mobile horizontal overflow was found. Fixed-world native contact sheets likewise
showed no foot crossing, collapse, or continuous unintended turn. Squat + bow reads
as a deep forward squat/bow; Pitch strafe travels visibly left while keeping its
orientation; and the restored Tall extension travels backward with a coherent strong
back extension. Tall is deliberately lower and more knee-bent than its v8 version.

## Native catalog and perturbation screen

Catalog 3 was retuned around the v8 catalog's pose and direction identities. Basic
forward/backward/left-strafe commands remain pure-axis; styled presets were allowed
small visible planar compensations but no hidden yaw-rate command. A focused
12,000-command Squat + bow search replaced the provisional candidate with
`[1.30,-0.05,0,0.33,0.05,0.72,0.25]`. All 12 final nominal motions were stable.
Across the nominal catalog:

- normalized tracking MAE: `0.015625`
- jitter: `0.014362`
- zero-turn-rate hold heading drift, mean / maximum: `1.858 / 5.282 deg`
- straight-motion semantic cross-track displacement, mean / maximum:
  `0.1047 / 0.2234 m`

Each final preset was then replayed from the nominal reset plus ten matched
perturbations. All `132/132` rollouts remained stable. Across those rollouts,
normalized tracking MAE was `0.015746` on average and `0.051258` in the worst
rollout; zero-turn-rate heading drift was `2.205 / 10.519 deg` mean / worst.

The three user-critical identity checks were screened separately with their own
direction/pose gates. Squat + bow passed `11/11` clear-forward/straight checks
(heading `1.53 / 4.07 deg` mean/worst; path-angle error `3.92 / 6.09 deg`). Pitch
strafe passed `11/11` leftward visual-path checks (heading `2.47 / 3.60 deg`;
path-angle error `1.05 / 2.51 deg`). Lowered Tall extension passed `11/11`
backward/back-extension checks (heading `2.70 / 10.52 deg`; path-angle error
`1.46 / 8.45 deg`). These are deterministic native MuJoCo screens and do not replace
Isaac Sim or hardware validation.

The local-only QA hook is
`?qaPreset=<preset-key>&qaTicks=400&v=v9-rhe0p5-240000-catalog3`; it writes release,
policy, initial-probe, and final-replay data to the hidden `#runtime-probe` element
for automation.
