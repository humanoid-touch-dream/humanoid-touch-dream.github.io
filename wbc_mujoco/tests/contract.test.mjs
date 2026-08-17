import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { HtdBrowserController } from "../src/controller.js";
import {
  buildObservation,
  clampCommand,
  projectedGravityFromWxyz,
  rangeBand,
  smootherstep,
  updateArmTargets,
} from "../src/contract.js";

const metadata = JSON.parse(
  await readFile(new URL("../public/assets/contract.json", import.meta.url), "utf8"),
);

test("browser metadata keeps the native dimensions and timing", () => {
  assert.equal(metadata.joint_names.length, 29);
  assert.equal(metadata.default_joint_pos.length, 29);
  assert.equal(metadata.kp.length, 29);
  assert.equal(metadata.kd.length, 29);
  assert.equal(metadata.effort_limit.length, 29);
  assert.equal(metadata.sim_dt, 0.005);
  assert.equal(metadata.control_decimation, 4);
  assert.equal(metadata.control_dt, 0.02);
  assert.equal(metadata.contact_history_steps, 3);
});

test("interactive input timing is separate from validated presets", () => {
  assert.equal(metadata.preset_settle_s, 1.0);
  assert.equal(metadata.preset_command_ramp_s, 2.0);
  assert.equal(metadata.slider_command_ramp_s, 0.2);
  assert.equal(metadata.keyboard_command_ramp_s, 0.0);
  assert.equal(smootherstep(0.5), 0.5);
});

test("keyboard commands apply on the next tick while sliders retain a short ramp", () => {
  const controller = Object.create(HtdBrowserController.prototype);
  controller.metadata = metadata;
  controller.command = Float32Array.from(metadata.neutral_command);
  controller.commandTarget = Float32Array.from(metadata.neutral_command);

  const keyboardTarget = Float32Array.from(metadata.neutral_command);
  keyboardTarget[0] = 0.1;
  controller.setManualTarget(keyboardTarget, { rampSeconds: metadata.keyboard_command_ramp_s });
  controller._advanceCommandPlan();
  assert.ok(Math.abs(controller.command[0] - 0.1) < 1e-7);
  assert.equal(controller.commandPlan, null);

  const sliderTarget = Float32Array.from(metadata.neutral_command);
  controller.setManualTarget(sliderTarget);
  assert.equal(controller.commandPlan.ramp, 0.2);
  for (let tick = 0; tick < 6; tick++) controller._advanceCommandPlan();
  assert.ok(Math.abs(controller.command[0] - 0.05) < 1e-6);
});

test("command clamp uses the Isaac play limits", () => {
  const value = clampCommand([-99, 99, -99, -1, 99, -99, 99], metadata);
  const expected = [-1.5, 1.5, -2.57, 0.3, 0.97, -0.9, 3.17];
  value.forEach((item, index) => assert.ok(Math.abs(item - expected[index]) < 1e-6));
});

test("every slider grid contains the neutral command exactly", () => {
  metadata.commands.forEach((spec, index) => {
    const gridIndex = (metadata.neutral_command[index] - spec.min) / spec.step;
    assert.ok(Math.abs(gridIndex - Math.round(gridIndex)) < 1e-9, spec.key);
  });
});

test("training envelopes are finite and map onto the play sliders", () => {
  const expected = [
    [-0.5, 0.5], [-0.5, 0.5], [-1.57, 1.57], [0.35, 0.8],
    [-0.7, 0.7], [-0.52, 1.57], [-1.57, 1.57],
  ];
  const expectedStart = [null, null, null, [0.5, 0.8], [-0.4, 0.4], [-0.3, 1.0], [-0.65, 0.65]];
  assert.deepEqual(metadata.commands.map((spec) => [spec.training.min, spec.training.max]), expected);
  assert.deepEqual(
    metadata.commands.map((spec) => spec.training.start_min === undefined
      ? null : [spec.training.start_min, spec.training.start_max]),
    expectedStart,
  );
  const bands = metadata.commands.map((spec) =>
    rangeBand(spec.min, spec.max, spec.training.min, spec.training.max));
  for (const band of bands) {
    assert.ok(Number.isFinite(band.leftPercent));
    assert.ok(Number.isFinite(band.widthPercent));
    assert.ok(band.leftPercent >= 0 && band.leftPercent <= 100);
    assert.ok(band.widthPercent >= 0 && band.leftPercent + band.widthPercent <= 100 + 1e-9);
  }
  assert.deepEqual(bands.map((band) => band.clippedLow), Array(7).fill(false));
  assert.deepEqual(bands.map((band) => band.clippedHigh), [false, false, false, false, false, true, false]);
});

test("projected gravity matches identity and a pure yaw", () => {
  const identity = projectedGravityFromWxyz([1, 0, 0, 0]);
  assert.ok(Math.abs(identity[0]) < 1e-7);
  assert.ok(Math.abs(identity[1]) < 1e-7);
  assert.equal(identity[2], -1);
  const half = Math.PI / 8;
  const yaw = projectedGravityFromWxyz([Math.cos(half), 0, 0, Math.sin(half)]);
  assert.ok(Math.abs(yaw[0]) < 1e-7);
  assert.ok(Math.abs(yaw[1]) < 1e-7);
  assert.ok(Math.abs(yaw[2] + 1) < 1e-7);
});

test("observation follows the exact 60-value order", () => {
  const jointPosition = Float64Array.from(metadata.default_joint_pos);
  jointPosition[0] += 0.125;
  const observation = buildObservation({
    angularVelocity: new Float64Array([1, 2, 3]),
    rootQuaternion: new Float64Array([1, 0, 0, 0]),
    command: new Float32Array(metadata.neutral_command),
    jointPosition,
    jointVelocity: new Float64Array(29),
    lastAction: new Float32Array(15),
    feetContact: new Float32Array([1, 0]),
    defaultJointPosition: metadata.default_joint_pos,
  });
  assert.equal(observation.length, 60);
  assert.deepEqual(Array.from(observation.slice(0, 3)), [1, 2, 3]);
  assert.ok(Math.abs(observation[3]) < 1e-7);
  assert.ok(Math.abs(observation[4]) < 1e-7);
  assert.equal(observation[5], -1);
  assert.ok(Math.abs(observation[13] - 0.125) < 1e-7);
  assert.deepEqual(Array.from(observation.slice(58)), [1, 0]);
});

test("arm targets are rate limited and frozen wrists stay fixed", () => {
  const currentTargets = Float64Array.from(metadata.arm_coupling.bias, (value, index) =>
    Math.min(metadata.arm_coupling.hi[index], Math.max(metadata.arm_coupling.lo[index], value)),
  );
  const next = updateArmTargets({
    metadata,
    command: new Float32Array([0.3, 0, 0, 0.4, 0, 0.8, 0]),
    jointPosition: Float64Array.from(metadata.default_joint_pos),
    currentTargets,
  });
  for (let index = 0; index < 14; index++) {
    assert.ok(Math.abs(next[index] - currentTargets[index]) <= 0.0600001);
  }
  for (const frozen of [5, 6, 12, 13]) assert.equal(next[frozen], 0);
});

test("smootherstep has exact endpoints and midpoint", () => {
  assert.equal(smootherstep(0), 0);
  assert.equal(smootherstep(0.5), 0.5);
  assert.equal(smootherstep(1), 1);
});
