# humanoid-touch-dream.github.io

This repository contains the source code for the Humanoid TouchDream website: https://humanoid-touch-dream.github.io/.

## Run locally

Build and verify the browser MuJoCo demo first:

```bash
cd wbc_mujoco
npm ci
npm test
npm run build
cd ..
```

Then serve the project root (opening the HTML directly will not load the WASM
and model assets correctly):

```bash
python3 -m http.server 8000
```

Open `http://127.0.0.1:8000/` and select **Launch interactive demo** in the
Whole-Body Controller section. The generated `wbc_mujoco/dist/` directory is
the production bundle published with this site. Rebuild it whenever the browser
demo source or assets change.

When this repository is beside `IsaacLab-Decoupled-WBC` in the HTD workspace,
`npm run check` additionally verifies the copied model, policy, and controller
contract against the native `sim2mujoco` implementation.

The production bundle includes the MuJoCo model, HTD policy, and the dependency
notices listed in `wbc_mujoco/public/licenses/THIRD_PARTY_NOTICES.txt`.
