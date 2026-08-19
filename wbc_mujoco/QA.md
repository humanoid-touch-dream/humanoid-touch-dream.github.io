# Local QA record

Checked on 2026-08-18 with Google Chrome headless/SwiftShader.

## Static and contract checks

- `npm run check`: pass (12 JavaScript contract tests, native controller/model-asset parity,
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
- browser policy: `ffw0p175_ft10k_ci20-60_wrp150-4_hpr150-3_jtl100_e12288_s1_v8`
  teacher at iteration `212500`; checkpoint SHA-256 `f66484126106476183354cf964ff24afce104ca1a1bd707c5cc84cf0f15c33d5`
  and ONNX SHA-256 `5d6d00ecbca33bb40f59eb76e47fb4688f7173b2576de9cca804c7492faaed82`
  match the declared contract provenance
- the webpage policy is intentionally independent from the sibling WBC repository's
  bundled v7 teacher/student/native-policy example
- pinned stack: `mujoco-js@0.0.7`, `onnxruntime-web@1.21.1`,
  `three@0.181.0`, `vite@6.4.3`

## Runtime parity probe

- browser MuJoCo runtime: 3.3.8
- native Python MuJoCo runtime: 3.3.7
- initial 60-value observation: exact match
- initial 15-value policy action from the same webpage ONNX under CPU/WASM:
  maximum absolute difference
  `7.152557373046875e-7` (mean `1.4174729301430489e-7`)
- production build loaded the model, rendered the robot, initialized ONNX Runtime,
  and completed the query-driven QA replay

The one-patch MuJoCo engine difference is intentional and documented; it is why the
browser stability screen below is separate from native validation.

## Browser stability screen

Every validated preset ran 400 policy ticks / 8 simulated seconds using the one-second
neutral settle and two-second smootherstep ramp. None triggered the fall detector.

| Preset | Min pelvis z | Final pelvis z | Final contact L/R | Fell |
|---|---:|---:|:---:|:---:|
| forward | 0.644295 | 0.672969 | 0 / 1 | no |
| backward | 0.644295 | 0.682761 | 1 / 1 | no |
| strafe_left | 0.644295 | 0.685768 | 0 / 1 | no |
| turn_left | 0.644295 | 0.663650 | 1 / 0 | no |
| squat_bow_walk | 0.376307 | 0.383501 | 1 / 0 | no |
| twisted_walk | 0.495603 | 0.501828 | 1 / 0 | no |
| side_lean_walk | 0.447966 | 0.454352 | 1 / 0 | no |
| pitch_strafe | 0.358421 | 0.369653 | 0 / 1 | no |
| forward_backbend | 0.582257 | 0.660779 | 1 / 0 | no |
| backlook_reverse | 0.644295 | 0.674012 | 1 / 0 | no |
| spin_backbend | 0.637819 | 0.697882 | 1 / 1 | no |
| tall_extension | 0.644295 | 0.745867 | 1 / 1 | no |

The revised `forward_backbend` command is `[0.8, 0, 0, 0.55, 0, -0.5, 0]`;
it removes the previous uncommanded curved path while retaining a clear backbend.

The local-only QA hook is `?qaPreset=<preset-key>&qaTicks=400`; it writes the initial
probe and final replay summary to the hidden `#runtime-probe` element for automation.
