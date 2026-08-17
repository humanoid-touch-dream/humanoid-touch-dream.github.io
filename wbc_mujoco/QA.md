# Local QA record

Checked on 2026-08-17 with Google Chrome headless/SwiftShader.

## Static and contract checks

- `npm run check`: pass (10 JavaScript contract tests, native metadata/asset sync,
  production Vite build)
- interaction timing is separated by source: keyboard `0.00 s`, sliders `0.20 s`,
  validated presets `2.00 s` after their one-second neutral settle
- every slider's step grid contains its exact neutral command, preventing inactive
  axes from acquiring small browser-generated offsets
- checkpoint training ranges and curriculum starts match the saved `env.yaml`;
  normalized bands remain inside the play slider, with only positive pitch
  correctly marked as extending beyond play
- `npm audit --omit=dev`: 0 vulnerabilities
- copied assets: policy, two MJCF XMLs, and all 36 referenced meshes match the
  native `sim2mujoco` copies byte-for-byte
- pinned stack: `mujoco-js@0.0.7`, `onnxruntime-web@1.21.1`,
  `three@0.181.0`, `vite@6.4.3`

## Runtime parity probe

- browser MuJoCo runtime: 3.3.8
- native Python MuJoCo runtime: 3.3.7
- initial 60-value observation: exact match
- initial 15-value CPU/WASM policy action: maximum absolute difference
  `3.5762786865234375e-7` (mean `1.3013680775960285e-7`)
- production build loaded the model, rendered the robot, initialized ONNX Runtime,
  and completed the query-driven QA replay

The one-patch MuJoCo engine difference is intentional and documented; it is why the
browser stability screen below is separate from native validation.

## Browser stability screen

Every validated preset ran 400 policy ticks / 8 simulated seconds using the one-second
neutral settle and two-second smootherstep ramp. None triggered the fall detector.

| Preset | Min pelvis z | Final pelvis z | Fell |
|---|---:|---:|:---:|
| forward | 0.646113 | 0.673749 | no |
| backward | 0.646113 | 0.682370 | no |
| strafe_left | 0.646113 | 0.679034 | no |
| turn_left | 0.646113 | 0.661694 | no |
| squat_bow_walk | 0.372882 | 0.384770 | no |
| twisted_walk | 0.497418 | 0.502231 | no |
| side_lean_walk | 0.443624 | 0.453908 | no |
| pitch_strafe | 0.365643 | 0.375422 | no |
| forward_backbend | 0.592464 | 0.652126 | no |
| backlook_reverse | 0.646113 | 0.667523 | no |
| spin_backbend | 0.629689 | 0.667048 | no |
| tall_extension | 0.646113 | 0.752452 | no |

The local-only QA hook is `?qaPreset=<preset-key>&qaTicks=400`; it writes the initial
probe and final replay summary to the hidden `#runtime-probe` element for automation.
