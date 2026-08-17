import loadMujoco from "mujoco-js/dist/mujoco_wasm.js";
import * as ort from "onnxruntime-web/wasm";

import {
  buildObservation,
  clamp,
  clampCommand,
  projectedGravityFromWxyz,
  smootherstep,
  updateArmTargets,
} from "./contract.js";

function enumValue(value) {
  return value && typeof value === "object" && "value" in value ? value.value : value;
}

async function fetchChecked(url, kind = "arrayBuffer") {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load ${url} (${response.status}).`);
  return kind === "text" ? response.text() : response.arrayBuffer();
}

function ensureMemfsDirectory(mujoco, path) {
  const parts = path.split("/").filter(Boolean);
  let current = "";
  for (const part of parts) {
    current += `/${part}`;
    if (!mujoco.FS.analyzePath(current).exists) mujoco.FS.mkdir(current);
  }
}

async function stageModelAssets(mujoco, assetRoot, metadata, onProgress) {
  ensureMemfsDirectory(mujoco, "/working/meshes");
  const files = ["scene.xml", "g1_29dof.xml", ...metadata.mesh_files.map((name) => `meshes/${name}`)];
  let completed = 0;

  await Promise.all(files.map(async (relativePath) => {
    const isXml = relativePath.endsWith(".xml");
    const payload = await fetchChecked(new URL(relativePath, assetRoot), isXml ? "text" : "arrayBuffer");
    const memfsPath = `/working/${relativePath}`;
    mujoco.FS.writeFile(memfsPath, isXml ? payload : new Uint8Array(payload));
    completed += 1;
    onProgress?.(`Loading robot assets ${completed}/${files.length}`);
  }));
}

function cleanupModelAssets(mujoco, metadata) {
  const files = ["scene.xml", "g1_29dof.xml", ...metadata.mesh_files.map((name) => `meshes/${name}`)];
  for (const relativePath of files.reverse()) {
    const path = `/working/${relativePath}`;
    if (mujoco.FS.analyzePath(path).exists) mujoco.FS.unlink(path);
  }
  if (mujoco.FS.analyzePath("/working/meshes").exists) mujoco.FS.rmdir("/working/meshes");
  if (mujoco.FS.analyzePath("/working").exists) mujoco.FS.rmdir("/working");
}

function findNamedId(mujoco, model, objectType, name) {
  const id = mujoco.mj_name2id(model, enumValue(objectType), name);
  if (id < 0) throw new Error(`MuJoCo model is missing ${name}.`);
  return id;
}

function readIndexed(source, indices) {
  return Float64Array.from(indices, (index) => source[index]);
}

export class HtdBrowserController {
  static async create({ baseUrl, onProgress }) {
    const assetRoot = new URL("assets/", baseUrl);
    onProgress?.("Loading the HTD control contract…");
    const metadata = JSON.parse(await fetchChecked(new URL("contract.json", assetRoot), "text"));

    onProgress?.("Loading MuJoCo WebAssembly…");
    const mujoco = await loadMujoco();
    let model;
    try {
      await stageModelAssets(mujoco, assetRoot, metadata, onProgress);
      onProgress?.("Compiling the G1 MuJoCo model…");
      model = mujoco.MjModel.loadFromXML("/working/scene.xml");
    } finally {
      cleanupModelAssets(mujoco, metadata);
    }
    const data = new mujoco.MjData(model);

    onProgress?.("Loading the HTD ONNX policy…");
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    ort.env.wasm.wasmPaths = new URL("vendor/ort/", baseUrl).href;
    const policy = await ort.InferenceSession.create(new URL("policy.onnx", assetRoot).href, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });

    return new HtdBrowserController({ mujoco, model, data, policy, metadata });
  }

  constructor({ mujoco, model, data, policy, metadata }) {
    this.mujoco = mujoco;
    this.model = model;
    this.data = data;
    this.policy = policy;
    this.metadata = metadata;
    this.inputName = policy.inputNames[0];
    this.outputName = policy.outputNames[0];

    this.lastAction = new Float32Array(metadata.action_dim);
    this.targetJointPosition = Float64Array.from(metadata.default_joint_pos);
    this.command = Float32Array.from(metadata.neutral_command);
    this.commandTarget = Float32Array.from(metadata.neutral_command);
    this.commandPlan = null;
    this.contactHistory = Array.from(
      { length: metadata.contact_history_steps },
      () => new Float32Array(2),
    );

    this._patchModel();
    this._resolveContract();
    this.reset();
  }

  get version() {
    if (typeof this.mujoco.mj_versionString === "function") return this.mujoco.mj_versionString();
    if (typeof this.mujoco.mj_version === "function") return String(this.mujoco.mj_version());
    return "unknown";
  }

  _patchModel() {
    const { metadata, model, mujoco } = this;
    model.opt.timestep = metadata.sim_dt;
    const jointObject = mujoco.mjtObj.mjOBJ_JOINT;
    for (const jointName of metadata.joint_names) {
      const jointId = findNamedId(mujoco, model, jointObject, jointName);
      const dofId = model.jnt_dofadr[jointId];
      model.dof_damping[dofId] = 0;
      model.dof_frictionloss[dofId] = 0;
      model.dof_armature[dofId] = 0.01;
    }

    const floorId = findNamedId(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, "floor");
    model.geom_contype[floorId] = 0;
    model.geom_conaffinity[floorId] = 1;
    this._setGeomFriction(floorId, metadata.floor_friction);
    for (let geomId = 0; geomId < model.ngeom; geomId++) {
      if (geomId === floorId) continue;
      if (model.geom_contype[geomId] || model.geom_conaffinity[geomId]) {
        model.geom_contype[geomId] = 1;
        model.geom_conaffinity[geomId] = 0;
        this._setGeomFriction(geomId, metadata.floor_friction);
      }
    }
  }

  _setGeomFriction(geomId, sliding) {
    const offset = geomId * 3;
    this.model.geom_friction[offset] = sliding;
    this.model.geom_friction[offset + 1] = 0.005;
    this.model.geom_friction[offset + 2] = 0.0001;
  }

  _resolveContract() {
    const { metadata, model, mujoco } = this;
    this.pelvisBodyId = findNamedId(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, metadata.body_names.pelvis);
    this.torsoBodyId = findNamedId(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, metadata.body_names.torso);
    this.leftFootBodyId = findNamedId(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, metadata.body_names.left_foot);
    this.rightFootBodyId = findNamedId(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, metadata.body_names.right_foot);

    this.jointIds = new Int32Array(metadata.joint_names.length);
    this.qposIds = new Int32Array(metadata.joint_names.length);
    this.qvelIds = new Int32Array(metadata.joint_names.length);
    this.actuatorIds = new Int32Array(metadata.joint_names.length);

    metadata.joint_names.forEach((jointName, index) => {
      const jointId = findNamedId(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, jointName);
      this.jointIds[index] = jointId;
      this.qposIds[index] = model.jnt_qposadr[jointId];
      this.qvelIds[index] = model.jnt_dofadr[jointId];

      let actuatorId = -1;
      for (let candidate = 0; candidate < model.nu; candidate++) {
        if (model.actuator_trnid[candidate * 2] === jointId) {
          if (actuatorId >= 0) throw new Error(`Multiple actuators drive ${jointName}.`);
          actuatorId = candidate;
        }
      }
      if (actuatorId < 0) throw new Error(`No actuator drives ${jointName}.`);
      this.actuatorIds[index] = actuatorId;

      const effort = metadata.effort_limit[index];
      model.actuator_ctrllimited[actuatorId] = 1;
      model.actuator_ctrlrange[actuatorId * 2] = -effort;
      model.actuator_ctrlrange[actuatorId * 2 + 1] = effort;
      model.jnt_actfrclimited[jointId] = 1;
      model.jnt_actfrcrange[jointId * 2] = -effort;
      model.jnt_actfrcrange[jointId * 2 + 1] = effort;
    });
  }

  reset() {
    const { data, metadata, model, mujoco } = this;
    mujoco.mj_resetData(model, data);
    data.qpos.set(model.qpos0);
    data.qpos[0] = 0;
    data.qpos[1] = 0;
    data.qpos[2] = 0.8;
    data.qpos[3] = 1;
    data.qpos[4] = 0;
    data.qpos[5] = 0;
    data.qpos[6] = 0;
    for (let index = 0; index < this.qposIds.length; index++) {
      data.qpos[this.qposIds[index]] = metadata.default_joint_pos[index];
    }
    data.qvel.fill(0);
    data.ctrl.fill(0);
    this.lastAction.fill(0);
    this.targetJointPosition.set(metadata.default_joint_pos);
    for (let arm = 0; arm < 14; arm++) {
      this.targetJointPosition[metadata.action_dim + arm] = clamp(
        metadata.arm_coupling.bias[arm],
        metadata.arm_coupling.lo[arm],
        metadata.arm_coupling.hi[arm],
      );
    }
    this.command.set(metadata.neutral_command);
    this.commandTarget.set(metadata.neutral_command);
    this.commandPlan = null;
    for (const contact of this.contactHistory) contact.fill(0);
    mujoco.mj_forward(model, data);
  }

  beginValidatedPreset(target) {
    const clamped = clampCommand(target, this.metadata);
    this.reset();
    this.commandTarget.set(clamped);
    this.commandPlan = {
      kind: "preset",
      elapsed: 0,
      start: Float32Array.from(this.metadata.neutral_command),
      target: clamped,
      settle: this.metadata.preset_settle_s,
      ramp: this.metadata.preset_command_ramp_s,
    };
  }

  setManualTarget(target, { rampSeconds = this.metadata.slider_command_ramp_s } = {}) {
    if (!Number.isFinite(rampSeconds) || rampSeconds < 0) {
      throw new Error("Manual command ramp must be a finite, non-negative duration.");
    }
    const clamped = clampCommand(target, this.metadata);
    this.commandTarget.set(clamped);
    this.commandPlan = {
      kind: "manual",
      elapsed: 0,
      start: Float32Array.from(this.command),
      target: clamped,
      settle: 0,
      ramp: rampSeconds,
    };
  }

  _advanceCommandPlan() {
    if (!this.commandPlan) return;
    const plan = this.commandPlan;
    let phase = 0;
    if (plan.elapsed >= plan.settle) {
      phase = plan.ramp > 0 ? smootherstep((plan.elapsed - plan.settle) / plan.ramp) : 1;
    }
    for (let axis = 0; axis < this.command.length; axis++) {
      this.command[axis] = plan.start[axis] + phase * (plan.target[axis] - plan.start[axis]);
    }
    plan.elapsed += this.metadata.control_dt;
    if (plan.elapsed >= plan.settle + plan.ramp) {
      this.command.set(plan.target);
      this.commandPlan = null;
    }
  }

  _feetContact() {
    const result = new Float32Array(2);
    for (const sample of this.contactHistory) {
      result[0] = Math.max(result[0], sample[0]);
      result[1] = Math.max(result[1], sample[1]);
    }
    return result;
  }

  _sampleFeetContact() {
    this.mujoco.mj_rnePostConstraint(this.model, this.data);
    const wrench = this.data.cfrc_ext;
    const forceNorm = (bodyId) => {
      const offset = bodyId * 6 + 3;
      return Math.hypot(wrench[offset], wrench[offset + 1], wrench[offset + 2]);
    };
    const sample = new Float32Array([
      forceNorm(this.leftFootBodyId) > this.metadata.contact_force_threshold ? 1 : 0,
      forceNorm(this.rightFootBodyId) > this.metadata.contact_force_threshold ? 1 : 0,
    ]);
    this.contactHistory.shift();
    this.contactHistory.push(sample);
  }

  observation() {
    return buildObservation({
      angularVelocity: this.data.qvel.subarray(3, 6),
      rootQuaternion: this.data.qpos.subarray(3, 7),
      command: this.command,
      jointPosition: readIndexed(this.data.qpos, this.qposIds),
      jointVelocity: readIndexed(this.data.qvel, this.qvelIds),
      lastAction: this.lastAction,
      feetContact: this._feetContact(),
      defaultJointPosition: this.metadata.default_joint_pos,
      clip: this.metadata.observation_clip,
    });
  }

  async initialProbe() {
    const observation = this.observation();
    const feeds = {
      [this.inputName]: new ort.Tensor("float32", observation, [1, this.metadata.observation_dim]),
    };
    const result = await this.policy.run(feeds);
    return {
      observation: Array.from(observation),
      action: Array.from(result[this.outputName].data),
    };
  }

  async _infer() {
    const observation = this.observation();
    const feeds = {
      [this.inputName]: new ort.Tensor("float32", observation, [1, this.metadata.observation_dim]),
    };
    const result = await this.policy.run(feeds);
    const rawAction = result[this.outputName].data;
    if (rawAction.length !== this.metadata.action_dim) {
      throw new Error(`Policy returned ${rawAction.length} actions, expected ${this.metadata.action_dim}.`);
    }

    for (let index = 0; index < rawAction.length; index++) {
      if (!Number.isFinite(rawAction[index])) throw new Error("Policy returned a non-finite action.");
      const action = clamp(rawAction[index], -this.metadata.action_clip, this.metadata.action_clip);
      this.lastAction[index] = action;
      this.targetJointPosition[index] =
        this.metadata.default_joint_pos[index] + this.metadata.action_scale * action;
    }

    const jointPosition = readIndexed(this.data.qpos, this.qposIds);
    const arms = updateArmTargets({
      metadata: this.metadata,
      command: this.command,
      jointPosition,
      currentTargets: this.targetJointPosition.subarray(this.metadata.action_dim),
    });
    this.targetJointPosition.set(arms, this.metadata.action_dim);
  }

  _physicsStep() {
    const { data, metadata } = this;
    for (let index = 0; index < this.qposIds.length; index++) {
      const position = data.qpos[this.qposIds[index]];
      const velocity = data.qvel[this.qvelIds[index]];
      const rawTorque = metadata.kp[index] * (this.targetJointPosition[index] - position)
        - metadata.kd[index] * velocity;
      const effort = metadata.effort_limit[index];
      data.ctrl[this.actuatorIds[index]] = clamp(rawTorque, -effort, effort);
    }
    this.mujoco.mj_step(this.model, data);
    this._sampleFeetContact();
  }

  async step() {
    this._advanceCommandPlan();
    await this._infer();
    for (let step = 0; step < this.metadata.control_decimation; step++) this._physicsStep();
    return this.diagnostics();
  }

  diagnostics() {
    const gravity = projectedGravityFromWxyz(this.data.qpos.subarray(3, 7));
    const finite = [this.data.qpos[0], this.data.qpos[1], this.data.qpos[2], ...gravity]
      .every(Number.isFinite);
    const fallen = !finite
      || this.data.qpos[2] < this.metadata.fall_pelvis_height
      || gravity[2] > this.metadata.fall_projected_gravity_z;
    return {
      time: this.data.time,
      rootPosition: this.data.qpos.subarray(0, 3),
      projectedGravity: gravity,
      feetContact: this._feetContact(),
      command: Float32Array.from(this.command),
      fallen,
    };
  }

  dispose() {
    this.policy?.release?.();
    this.data?.delete?.();
    this.model?.delete?.();
    this.data = null;
    this.model = null;
  }
}
