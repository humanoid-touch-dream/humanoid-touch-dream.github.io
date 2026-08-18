# Local QA record

Checked on 2026-08-18 with Google Chrome headless/SwiftShader.

## Static and contract checks

- `npm run check`: pass (11 JavaScript contract tests, native metadata/asset sync,
  production Vite build)
- interaction timing is separated by source: keyboard `0.00 s`, sliders `0.20 s`,
  validated presets `2.00 s` after their one-second neutral settle
- every slider's step grid contains its exact neutral command, preventing inactive
  axes from acquiring small browser-generated offsets
- checkpoint training ranges and curriculum starts match the saved `env.yaml`;
  normalized bands remain inside the play slider, with only positive pitch
  correctly marked as extending beyond play
- `npm audit --omit=dev`: 0 vulnerabilities
- the two MJCF XMLs and all 36 referenced meshes match the native `sim2mujoco`
  copies byte-for-byte
- browser policy: `ffw0p1_ft10k_ci20-60_wrp150-4_hpr150-3_jtl100_e12288_s1_v8`
  teacher at iteration `211500`; checkpoint SHA-256 `c16bcee71a7bddec4567b29cf18ef8568cc7dfaac637a57c0de0ccb66bd3d951`
  and ONNX SHA-256 `432d16fbc5b579849924af815be8678f04d20c38ff4ff61318186fdfc6fb7e3e`
  match the declared contract provenance
- pinned stack: `mujoco-js@0.0.7`, `onnxruntime-web@1.21.1`,
  `three@0.181.0`, `vite@6.4.3`

## Runtime parity probe

- browser MuJoCo runtime: 3.3.8
- native Python MuJoCo runtime: 3.3.7
- initial 60-value observation: exact match
- initial 15-value CPU/WASM policy action: maximum absolute difference
  `5.960464477539062e-7` (mean `1.7831722232131142e-7`)
- production build loaded the model, rendered the robot, initialized ONNX Runtime,
  and completed the query-driven QA replay

The one-patch MuJoCo engine difference is intentional and documented; it is why the
browser stability screen below is separate from native validation.

## Browser stability screen

Every validated preset ran 400 policy ticks / 8 simulated seconds using the one-second
neutral settle and two-second smootherstep ramp. None triggered the fall detector.

| Preset | Min pelvis z | Final pelvis z | Fell |
|---|---:|---:|:---:|
| forward | 0.649433 | 0.680893 | no |
| backward | 0.649433 | 0.681333 | no |
| strafe_left | 0.649433 | 0.687907 | no |
| turn_left | 0.649433 | 0.667414 | no |
| squat_bow_walk | 0.376175 | 0.394725 | no |
| twisted_walk | 0.508815 | 0.514664 | no |
| side_lean_walk | 0.458292 | 0.467748 | no |
| pitch_strafe | 0.358570 | 0.376484 | no |
| forward_backbend | 0.604733 | 0.627112 | no |
| backlook_reverse | 0.649433 | 0.668325 | no |
| spin_backbend | 0.644357 | 0.696521 | no |
| tall_extension | 0.649433 | 0.747707 | no |

The local-only QA hook is `?qaPreset=<preset-key>&qaTicks=400`; it writes the initial
probe and final replay summary to the hidden `#runtime-probe` element for automation.
