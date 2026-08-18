# HTD WBC interactive browser demo

This directory contains a standalone browser port of the native
`IsaacLab-Decoupled-WBC/sim2mujoco` contract. It runs MuJoCo WebAssembly and the
exported ONNX policy in the browser; it does not connect to Isaac Sim or a server.

## Run locally

```bash
npm ci
npm test
npm run dev
```

Open the URL printed by Vite, normally `http://127.0.0.1:5173/`. The build uses
relative URLs and can later be embedded in the HTD project page as a static iframe.

In the full HTD workspace, `npm run check` also compares the browser contract and
copied robot assets against the sibling `IsaacLab-Decoupled-WBC/sim2mujoco` tree,
checks the browser policy against its declared hash, then produces the local
`dist/` build. A standalone website checkout can use `npm test && npm run build`
without that sibling parity source.

## Parity contract

The current browser policy is the teacher actor from
`ffw0p1_ft10k_ci20-60_wrp150-4_hpr150-3_jtl100_e12288_s1_v8` at iteration
`211500`, fine-tuned with flat-foot reward weight `0.10`. Its source-checkpoint
and exported-ONNX hashes are pinned in `public/assets/contract.json`.

- MuJoCo physics: 200 Hz (`dt=0.005`)
- policy/control: 50 Hz (four physics steps per inference)
- 60 observations to 15 actions, action scale `0.25`
- explicit PD control with native gains and effort clipping
- zero joint damping/friction loss, armature `0.01`, robot self-collision disabled
- two foot-contact bits using a three-physics-step force history at `0.5 N`
- coupled, rate-limited arms from the native exported fit
- validated preset buttons reset, settle at neutral for one second, then use the
  native validator's two-second smootherstep transition
- keyboard steps are applied on the next 50 Hz control tick; slider changes use
  a short 0.20-second transition to avoid abrupt large command jumps
- play-limit sliders with checkpoint training envelopes shown as teal bands

Only presets that passed the current native MuJoCo screen are exposed as named
presets. The sliders deliberately retain the full Isaac **play** ranges, so arbitrary
slider combinations are exploratory rather than validated.

The compact legend identifies the teal bands as the checkpoint's final configured
training envelope. Exact per-axis ranges and the iteration 20,000–60,000 posture
curriculum remain available to assistive technology and as range-map tooltips.
Torso pitch is an important exception to the usual wider-play pattern: training
reached `+1.57` rad while the play slider stops at `+1.27` rad, so its band ends
with an overflow arrow.

The published bundle contains the model, meshes, ONNX policy, and metadata needed
to run without a server. Attribution and redistribution notices are documented in
`THIRD_PARTY.md` and bundled at `public/licenses/THIRD_PARTY_NOTICES.txt`.

`mujoco-js` 0.0.7 is intentionally pinned because it is the same browser binding used
by the SceneBot reference and exposes the model/data arrays required by this implementation.
Its embedded MuJoCo reports **3.3.8**, adjacent to (but not identical with) the native
Python MuJoCo 3.3.7 validator. This version difference must remain explicit, and the
browser trace/stability checks must pass independently before publishing. A future move
to `@mujoco/mujoco` should likewise replay those checks.
