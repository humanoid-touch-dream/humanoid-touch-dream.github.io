import { HtdBrowserController } from "./controller.js";
import { clampCommand, rangeBand } from "./contract.js";
import { HtdThreeRenderer } from "./renderer.js";

const elements = {
  viewport: document.querySelector("#viewport"),
  loading: document.querySelector("#loading"),
  loadingMessage: document.querySelector("#loading-message"),
  status: document.querySelector("#status"),
  statusDot: document.querySelector("#status-dot"),
  simTime: document.querySelector("#sim-time"),
  contacts: document.querySelector("#contacts"),
  start: document.querySelector("#start"),
  pause: document.querySelector("#pause"),
  reset: document.querySelector("#reset"),
  neutral: document.querySelector("#neutral"),
  presets: document.querySelector("#presets"),
  sliders: document.querySelector("#sliders"),
};

let controller;
let visualizer;
let running = false;
let timer = null;
let loopGeneration = 0;
let activePreset = null;
const sliderElements = [];
const presetButtons = new Map();

function formatRangeValue(value, showPositive) {
  if (Math.abs(value) < 5e-7) return "0.00";
  const magnitude = Math.abs(value).toFixed(2);
  if (value < 0) return `−${magnitude}`;
  return showPositive ? `+${magnitude}` : magnitude;
}

function trainingRangeDescription(spec) {
  const showPositive = spec.training.min < 0 || (spec.training.start_min ?? 0) < 0;
  const finalRange = `${formatRangeValue(spec.training.min, showPositive)}…${formatRangeValue(spec.training.max, showPositive)}`;
  if (spec.training.start_min === undefined) return `Training range ${finalRange} ${spec.unit}.`;
  const startRange = `${formatRangeValue(spec.training.start_min, showPositive)}…${formatRangeValue(spec.training.start_max, showPositive)}`;
  return `Training curriculum ${startRange} to ${finalRange} ${spec.unit}.`;
}

function updateSliderValueText(input, spec, value) {
  const insideTraining = value >= spec.training.min - 1e-7 && value <= spec.training.max + 1e-7;
  input.setAttribute(
    "aria-valuetext",
    `${Number(value).toFixed(2)} ${spec.unit}; ${insideTraining ? "inside" : "outside"} the final training range`,
  );
}

function setStatus(kind, text) {
  elements.statusDot.className = `status-dot ${kind}`;
  elements.status.textContent = text;
}

function setLoadingMessage(text) {
  elements.loadingMessage.textContent = text;
  setStatus("loading", text);
}

function enableControls(enabled) {
  for (const element of [elements.start, elements.pause, elements.reset, elements.neutral]) {
    element.disabled = !enabled;
  }
  for (const { input } of sliderElements) input.disabled = !enabled;
  for (const button of presetButtons.values()) button.disabled = !enabled;
}

function updateTransport() {
  elements.start.disabled = !controller || running;
  elements.pause.disabled = !controller || !running;
  elements.start.textContent = Number(controller?.data.time || 0) > 0 ? "Resume" : "Start";
}

function updatePresetHighlight() {
  for (const [key, button] of presetButtons) button.classList.toggle("active", key === activePreset);
}

function updateSliderUi(command) {
  sliderElements.forEach(({ input, output, spec }, index) => {
    input.value = String(command[index]);
    output.value = Number(command[index]).toFixed(2);
    output.textContent = Number(command[index]).toFixed(2);
    updateSliderValueText(input, spec, command[index]);
  });
}

function desiredCommandFromUi() {
  return clampCommand(sliderElements.map(({ input }) => Number(input.value)), controller.metadata);
}

function setManualCommandFromUi() {
  if (!controller) return;
  activePreset = null;
  updatePresetHighlight();
  controller.setManualTarget(desiredCommandFromUi());
}

function buildControls(metadata) {
  metadata.commands.forEach((spec, index) => {
    const row = document.createElement("div");
    row.className = "slider-row";
    const label = document.createElement("label");
    const sliderControl = document.createElement("div");
    sliderControl.className = "slider-control";
    const input = document.createElement("input");
    const output = document.createElement("output");
    const rangeMap = document.createElement("div");
    const trainingBand = document.createElement("span");
    const trainingDescription = document.createElement("span");
    const inputId = `command-${spec.key}`;
    const trainingId = `${inputId}-training-range`;
    label.htmlFor = inputId;
    label.textContent = spec.label;
    input.id = inputId;
    input.type = "range";
    input.min = String(spec.min);
    input.max = String(spec.max);
    input.step = String(spec.step);
    input.value = String(metadata.neutral_command[index]);
    input.disabled = true;
    output.value = Number(input.value).toFixed(2);
    output.textContent = output.value;
    output.title = spec.unit;
    output.setAttribute("for", inputId);

    const band = rangeBand(spec.min, spec.max, spec.training.min, spec.training.max);
    rangeMap.className = "range-map";
    rangeMap.classList.toggle("clipped-low", band.clippedLow);
    rangeMap.classList.toggle("clipped-high", band.clippedHigh);
    rangeMap.title = spec.training.start_min === undefined
      ? `Training range: ${spec.training.min} to ${spec.training.max} ${spec.unit}`
      : `Training curriculum: ${spec.training.start_min} to ${spec.training.start_max}, then ${spec.training.min} to ${spec.training.max} ${spec.unit}`;
    rangeMap.setAttribute("aria-hidden", "true");
    trainingBand.className = "training-band";
    trainingBand.style.setProperty("--training-left", `${band.leftPercent}%`);
    trainingBand.style.setProperty("--training-width", `${band.widthPercent}%`);
    rangeMap.append(trainingBand);

    trainingDescription.id = trainingId;
    trainingDescription.className = "sr-only";
    trainingDescription.textContent = trainingRangeDescription(spec);
    input.setAttribute("aria-describedby", `training-range-legend ${trainingId}`);
    updateSliderValueText(input, spec, input.value);
    input.addEventListener("input", () => {
      output.value = Number(input.value).toFixed(2);
      output.textContent = output.value;
      updateSliderValueText(input, spec, input.value);
      setManualCommandFromUi();
    });
    sliderControl.append(input, rangeMap);
    row.append(label, sliderControl, output, trainingDescription);
    elements.sliders.append(row);
    sliderElements.push({ input, output, spec });
  });

  metadata.safe_presets.forEach((preset) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = preset.label;
    button.disabled = true;
    button.addEventListener("click", () => runValidatedPreset(preset));
    elements.presets.append(button);
    presetButtons.set(preset.key, button);
  });
}

function updateHud(diagnostics) {
  const time = Number(diagnostics.time);
  elements.simTime.textContent = `${Number.isFinite(time) ? time.toFixed(2) : "0.00"} s`;
  const feet = diagnostics.feetContact;
  elements.contacts.textContent = `feet · ${Math.round(feet[0])} ${Math.round(feet[1])}`;

  if (diagnostics.fallen) {
    pauseSimulation(false);
    setStatus("fallen", "Fall detected · reset to continue");
  } else if (running) {
    const phase = controller.commandPlan?.kind === "preset"
      ? controller.commandPlan.elapsed < controller.metadata.preset_settle_s
        ? "settling at neutral"
        : "ramping command"
      : "running";
    setStatus("ready", `${phase} · MuJoCo ${controller.version}`);
  }
}

function startSimulation() {
  if (!controller || running) return;
  running = true;
  loopGeneration += 1;
  const generation = loopGeneration;
  const tickMilliseconds = controller.metadata.control_dt * 1000;
  let nextDeadline = performance.now() + tickMilliseconds;
  updateTransport();
  setStatus("ready", `running · MuJoCo ${controller.version}`);

  const tick = async () => {
    if (!running || generation !== loopGeneration) return;
    try {
      const diagnostics = await controller.step();
      visualizer.sync();
      updateHud(diagnostics);
      if (!running || generation !== loopGeneration) return;
    } catch (error) {
      console.error(error);
      pauseSimulation(false);
      setStatus("error", `Simulation error · ${error.message}`);
      return;
    }

    nextDeadline += tickMilliseconds;
    let delay = nextDeadline - performance.now();
    if (delay < -100) {
      nextDeadline = performance.now() + tickMilliseconds;
      delay = tickMilliseconds;
    }
    timer = window.setTimeout(tick, Math.max(0, delay));
  };
  timer = window.setTimeout(tick, tickMilliseconds);
}

function pauseSimulation(showStatus = true) {
  running = false;
  loopGeneration += 1;
  if (timer !== null) window.clearTimeout(timer);
  timer = null;
  updateTransport();
  if (showStatus && controller) setStatus("paused", `paused · MuJoCo ${controller.version}`);
}

function resetSimulation({ pause = true } = {}) {
  if (!controller) return;
  if (pause) pauseSimulation(false);
  controller.reset();
  visualizer.sync();
  activePreset = null;
  updatePresetHighlight();
  updateSliderUi(controller.metadata.neutral_command);
  updateHud(controller.diagnostics());
  if (pause) setStatus("paused", `reset · MuJoCo ${controller.version}`);
  updateTransport();
}

function runValidatedPreset(preset) {
  if (!controller) return;
  pauseSimulation(false);
  controller.beginValidatedPreset(preset.command);
  visualizer.sync();
  activePreset = preset.key;
  updatePresetHighlight();
  updateSliderUi(preset.command);
  updateHud(controller.diagnostics());
  startSimulation();
}

function handleKeyboard(event) {
  if (!controller || event.repeat) return;
  const target = Float32Array.from(controller.commandTarget);
  const key = event.key.toLowerCase();
  const increments = {
    w: [0, 0.1], s: [0, -0.1],
    a: [1, 0.1], d: [1, -0.1],
    q: [2, 0.1], e: [2, -0.1],
    h: [3, 0.01], j: [3, -0.01],
    z: [4, 0.1], x: [4, -0.1],
    c: [5, 0.1], v: [5, -0.1],
    b: [6, 0.1], n: [6, -0.1],
  };

  if (event.code === "Space") {
    running ? pauseSimulation() : startSimulation();
    event.preventDefault();
    return;
  }
  if (key === "r") {
    resetSimulation();
    event.preventDefault();
    return;
  }
  if (!(key in increments)) return;
  const [axis, delta] = increments[key];
  target[axis] += delta;
  controller.setManualTarget(target, {
    rampSeconds: controller.metadata.keyboard_command_ramp_s,
  });
  updateSliderUi(controller.commandTarget);
  activePreset = null;
  updatePresetHighlight();
  event.preventDefault();
}

async function boot() {
  try {
    const baseUrl = new URL(import.meta.env.BASE_URL, window.location.href);
    controller = await HtdBrowserController.create({ baseUrl, onProgress: setLoadingMessage });
    const initialProbe = await controller.initialProbe();
    const runtimeProbe = {
      mujocoVersion: controller.version,
      ...initialProbe,
    };
    buildControls(controller.metadata);
    visualizer = new HtdThreeRenderer({
      container: elements.viewport,
      model: controller.model,
      data: controller.data,
      pelvisBodyId: controller.pelvisBodyId,
    });
    enableControls(true);
    const query = new URLSearchParams(window.location.search);
    const qaPresetKey = query.get("qaPreset");
    const qaPreset = controller.metadata.safe_presets.find((preset) => preset.key === qaPresetKey);
    if (qaPreset) {
      const requestedTicks = Number.parseInt(query.get("qaTicks") || "400", 10);
      const qaTicks = Math.min(1000, Math.max(1, Number.isFinite(requestedTicks) ? requestedTicks : 400));
      setStatus("loading", `QA replay · ${qaPreset.label}`);
      controller.beginValidatedPreset(qaPreset.command);
      let diagnostics = controller.diagnostics();
      let rootHeightMin = Number.POSITIVE_INFINITY;
      let completedTicks = 0;
      const qaStarted = performance.now();
      for (; completedTicks < qaTicks && !diagnostics.fallen; completedTicks++) {
        diagnostics = await controller.step();
        rootHeightMin = Math.min(rootHeightMin, diagnostics.rootPosition[2]);
      }
      visualizer.sync();
      activePreset = qaPreset.key;
      updatePresetHighlight();
      updateSliderUi(qaPreset.command);
      updateHud(diagnostics);
      pauseSimulation(false);
      setStatus(
        diagnostics.fallen ? "fallen" : "paused",
        `QA ${diagnostics.fallen ? "failed" : "passed"} · ${qaPreset.label}`,
      );
      runtimeProbe.qa = {
        preset: qaPreset.key,
        requestedTicks: qaTicks,
        completedTicks,
        simulatedSeconds: diagnostics.time,
        wallMilliseconds: performance.now() - qaStarted,
        fallen: diagnostics.fallen,
        rootHeightMin,
        rootHeightFinal: diagnostics.rootPosition[2],
        feetContact: Array.from(diagnostics.feetContact),
        command: Array.from(diagnostics.command),
      };
    } else {
      resetSimulation();
    }
    document.querySelector("#runtime-probe").textContent = JSON.stringify(runtimeProbe);
    document.documentElement.dataset.htdWbcReady = "true";
    elements.loading.classList.add("hidden");
    window.dispatchEvent(new CustomEvent("htd-wbc:ready"));
    if (window.parent !== window) window.parent.postMessage({ type: "htd-wbc:ready" }, "*");
  } catch (error) {
    console.error(error);
    elements.loading.querySelector(".loader")?.remove();
    elements.loadingMessage.textContent = `Could not start the demo: ${error.message}`;
    setStatus("error", "Demo failed to load");
  }
}

elements.start.addEventListener("click", startSimulation);
elements.pause.addEventListener("click", () => pauseSimulation());
elements.reset.addEventListener("click", () => resetSimulation());
elements.neutral.addEventListener("click", () => {
  pauseSimulation(false);
  resetSimulation({ pause: false });
  startSimulation();
});
window.addEventListener("keydown", handleKeyboard);
window.addEventListener("pagehide", () => {
  pauseSimulation(false);
  visualizer?.dispose();
  controller?.dispose();
}, { once: true });

boot();
