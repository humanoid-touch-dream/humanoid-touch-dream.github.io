# Third-party notes

- Browser physics uses Google DeepMind's `mujoco-js` 0.0.7 package
  (Apache-2.0). Its embedded runtime reports MuJoCo 3.3.8; native validation
  currently uses 3.3.7, so browser results receive a separate stability screen.
- Neural inference uses ONNX Runtime Web 1.21.1 (MIT).
- Rendering uses Three.js 0.181.0 (MIT).
- The G1 MJCF and 36 meshes were copied through the HTD `sim2mujoco` layer from
  NVIDIA's Apache-2.0
  [GR00T Whole-Body Control](https://github.com/NVlabs/GR00T-WholeBodyControl)
  source tree. The model describes the Unitree G1 robot; Unitree names and
  product designs remain the property of Unitree Robotics.
- The ONNX file is a project-trained HTD policy, not an NVIDIA pretrained model
  weight, and is distributed under the HTD project's BSD 3-Clause license.

The production build includes the full applicable notices in
`public/licenses/THIRD_PARTY_NOTICES.txt`. The rendering code in this directory
is purpose-built for HTD and does not copy SceneBot's rendering utility.
