export function clamp(value, low, high) {
  return Math.min(high, Math.max(low, value));
}

export function smootherstep(value) {
  const x = clamp(value, 0, 1);
  return x * x * x * (x * (x * 6 - 15) + 10);
}

export function clampCommand(command, metadata) {
  if (!command || command.length !== metadata.commands.length) {
    throw new Error(`Expected ${metadata.commands.length} command values.`);
  }
  return Float32Array.from(command, (value, index) => {
    if (!Number.isFinite(value)) throw new Error("Command contains a non-finite value.");
    const spec = metadata.commands[index];
    return clamp(value, spec.min, spec.max);
  });
}

export function rangeBand(playMin, playMax, trainingMin, trainingMax) {
  const values = [playMin, playMax, trainingMin, trainingMax];
  if (!values.every(Number.isFinite) || playMax <= playMin || trainingMax < trainingMin) {
    throw new Error("Invalid play or training command range.");
  }
  const span = playMax - playMin;
  const visibleMin = clamp(trainingMin, playMin, playMax);
  const visibleMax = clamp(trainingMax, playMin, playMax);
  return {
    leftPercent: 100 * (visibleMin - playMin) / span,
    widthPercent: 100 * Math.max(0, visibleMax - visibleMin) / span,
    clippedLow: trainingMin < playMin,
    clippedHigh: trainingMax > playMax,
  };
}

export function projectedGravityFromWxyz(quaternion) {
  const [w, x, y, z] = quaternion;
  return new Float32Array([
    -2 * (x * z - w * y),
    -2 * (y * z + w * x),
    -(1 - 2 * (x * x + y * y)),
  ]);
}

export function buildObservation({
  angularVelocity,
  rootQuaternion,
  command,
  jointPosition,
  jointVelocity,
  lastAction,
  feetContact,
  defaultJointPosition,
  clip = 100,
}) {
  const observation = new Float32Array(60);
  let offset = 0;
  const append = (values) => {
    for (const value of values) observation[offset++] = clamp(value, -clip, clip);
  };

  append(angularVelocity);
  append(projectedGravityFromWxyz(rootQuaternion));
  append(command);
  for (let index = 0; index < 15; index++) {
    observation[offset++] = clamp(jointPosition[index] - defaultJointPosition[index], -clip, clip);
  }
  append(jointVelocity.subarray(0, 15));
  append(lastAction);
  append(feetContact);

  if (offset !== observation.length) throw new Error(`Observation has ${offset} values, expected 60.`);
  return observation;
}

export function updateArmTargets({ metadata, command, jointPosition, currentTargets }) {
  const coupling = metadata.arm_coupling;
  const regressors = [command[4], command[5], command[6], command[3] - metadata.arm_stand_height];
  const hipDifference = jointPosition[0] - jointPosition[6];
  const maxStep = metadata.arm_rate_limit * metadata.control_dt;
  const next = Float64Array.from(currentTargets);

  for (let arm = 0; arm < 14; arm++) {
    let desired = coupling.bias[arm] + hipDifference * coupling.swing[arm];
    for (let term = 0; term < 4; term++) desired += coupling.gain[arm][term] * regressors[term];
    desired = clamp(desired, coupling.lo[arm], coupling.hi[arm]);
    next[arm] += clamp(desired - next[arm], -maxStep, maxStep);
  }
  return next;
}
