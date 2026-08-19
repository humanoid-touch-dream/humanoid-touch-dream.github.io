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
- browser policy: `ffw0p2_ft10k_ci20-60_wrp150-4_hpr150-3_jtl100_e12288_s1_v8`
  teacher at iteration `210500`; checkpoint SHA-256 `408e2f5176a72cdd98bd59e7724ab6f11233cb18e3a8dabf099d49585f7eb063`
  and ONNX SHA-256 `a495c994a9ee33c4a7a0cbfdcef248287ac66503d2c178fe141d72846db85410`
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
  `5.960464477539062e-7` (mean `2.5530655989314255e-7`)
- production build loaded the model, rendered the robot, initialized ONNX Runtime,
  and completed the query-driven QA replay

The one-patch MuJoCo engine difference is intentional and documented; it is why the
browser stability screen below is separate from native validation.

## Browser stability screen

Every validated preset ran 400 policy ticks / 8 simulated seconds using the one-second
neutral settle and two-second smootherstep ramp. None triggered the fall detector.

| Preset | Min pelvis z | Final pelvis z | Final contact L/R | Fell |
|---|---:|---:|:---:|:---:|
| forward | 0.640094 | 0.670173 | 0 / 1 | no |
| backward | 0.640094 | 0.679588 | 1 / 0 | no |
| strafe_left | 0.640094 | 0.693577 | 0 / 1 | no |
| turn_left | 0.640094 | 0.666873 | 1 / 1 | no |
| squat_bow_walk | 0.370059 | 0.376241 | 1 / 0 | no |
| twisted_walk | 0.568245 | 0.570205 | 1 / 0 | no |
| side_lean_walk | 0.428349 | 0.437838 | 0 / 1 | no |
| pitch_strafe | 0.342167 | 0.345074 | 0 / 1 | no |
| forward_backbend | 0.563565 | 0.657980 | 1 / 0 | no |
| backlook_reverse | 0.534647 | 0.549430 | 1 / 1 | no |
| spin_backbend | 0.630074 | 0.632393 | 1 / 0 | no |
| tall_extension | 0.640094 | 0.744527 | 1 / 1 | no |

The revised `forward_backbend` command is `[0.7, 0, 0, 0.5, 0, -0.5, 0]`;
the production-browser replay retained a clear, stable backbend.

## Preset path screen

The final catalog was also compared with the previous commands in native MuJoCo
using the nominal reset plus eight matched perturbations per preset. Both catalogs
remained stable in all 108 trials. Across the ten zero-turn-rate presets, the tuned
catalog reduced mean absolute world-heading drift from `16.28` to `3.77` degrees,
mean path-angle error from `8.16` to `2.62` degrees, and mean cross-track displacement
from `0.347` to `0.117 m`. The tuned straight-moving presets all retain an exact
commanded turn rate of zero.

The local-only QA hook is `?qaPreset=<preset-key>&qaTicks=400`; it writes the initial
probe and final replay summary to the hidden `#runtime-probe` element for automation.
