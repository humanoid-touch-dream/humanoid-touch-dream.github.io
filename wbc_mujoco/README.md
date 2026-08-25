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
checks the independent webpage policy against its declared hash, then produces the local
`live/` build. A standalone website checkout can use `npm test && npm run build`
without that sibling parity source.

## Parity contract

The current browser policy is the teacher actor from
`rhe0p5_ft50k_ci20-60_wrp150-4_hpr150-3_fcs0p5_jtl100_e12288_s1_v9` at iteration
`240000`, fine-tuned with flat-foot reward weight `0.175`. Among non-standing
moving environments, its command mixture uses 50% heading-controlled environments;
among the remaining direct-yaw environments, 50% receive an exact zero turn-rate
command. It was selected for this browser showcase after screening every retained
v9 checkpoint: it improves common-grid and high-speed tracking, perturbation
performance, neutral full-sole contact, and broad path straightness over the
previous webpage policy. It is not universally stronger—the previous v8 policy
responds better to some low-speed and random two-axis in-distribution commands—so
the named catalog below is validated independently for this checkpoint. Its
source-checkpoint and exported-ONNX hashes are pinned in
`public/assets/contract.json`.
This policy is specific to the webpage demo. The sibling WBC repository keeps its
bundled teacher, student, and native MuJoCo example on v7 until a matching student
has been trained.

- MuJoCo physics: 200 Hz (`dt=0.005`)
- policy/control: 50 Hz (four physics steps per inference)
- 60 observations to 15 actions, action scale `0.25`
- explicit PD control with native gains and effort clipping
- zero joint damping/friction loss, armature `0.01`, robot self-collision disabled
- two foot-contact bits using a three-physics-step force history at `0.5 N`
- coupled, rate-limited arms from the native exported fit
- validated preset buttons reset, settle at neutral for one second, then use a
  two-second smootherstep transition
- keyboard steps are applied on the next 50 Hz control tick; slider commands use
  a per-tick slew limit, so dragging continuously updates the policy without
  restarting the transition; a full-range change takes ten control ticks
  (nominally `0.20 s`)
- play-limit sliders with checkpoint training envelopes shown as teal bands

The named catalog passed native MuJoCo stability/semantic screening and an
independent browser replay for this policy. The sliders deliberately retain the full
Isaac **play** ranges, so arbitrary slider combinations are exploratory rather than
validated.
The named zero-turn-rate locomotion presets were retuned specifically for this
checkpoint to reduce integrated world-heading drift and cross-track motion; they do
not rely on a hidden counter-turn command. Catalog 3 preserves the previous twelve
motion labels and their pose/direction identities instead of selecting arbitrary
poses solely for the lowest tracking score. Forward, backward, and left strafe use
pure-axis planar commands. Styled motions keep the same qualitative pose and travel
direction, with small, visible planar-command adjustments where needed to make their
achieved paths visually straighter. In particular, Squat + bow uses
`vy=-0.05 m/s`, and Pitch strafe uses `vx=+0.15 m/s`; the sliders show these values,
and there is no hidden heading controller. `Tall extension` is retained as a
backward walk with the torso height lowered from `0.85 m` to the neutral `0.72 m`.
Its label therefore refers to the strong back-extension pose rather than an elevated
torso target. Every preset command lies exactly on the corresponding browser slider
grid.

The compact legend identifies the teal bands as the checkpoint's final configured
training envelope. The bundled v7 example has the same command ranges and provides
the portable range/curriculum reference used by the parity check; it does not
describe this v9 checkpoint's heading/direct-yaw population mixture. Exact per-axis
ranges and the iteration 20,000–60,000 posture
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
