import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

function setMujocoPosition(buffer, index, target) {
  target.set(buffer[index * 3], buffer[index * 3 + 2], -buffer[index * 3 + 1]);
}

function setMujocoQuaternion(buffer, index, target) {
  const offset = index * 4;
  target.set(buffer[offset + 1], buffer[offset + 3], -buffer[offset + 2], buffer[offset]);
}

function rgbaForGeom(model, geomId) {
  const offset = geomId * 4;
  return [
    model.geom_rgba[offset],
    model.geom_rgba[offset + 1],
    model.geom_rgba[offset + 2],
    model.geom_rgba[offset + 3],
  ];
}

function meshGeometry(model, meshId) {
  const vertexStart = model.mesh_vertadr[meshId] * 3;
  const vertexCount = model.mesh_vertnum[meshId];
  const source = model.mesh_vert.subarray(vertexStart, vertexStart + vertexCount * 3);
  const positions = new Float32Array(source.length);
  for (let offset = 0; offset < source.length; offset += 3) {
    positions[offset] = source[offset];
    positions[offset + 1] = source[offset + 2];
    positions[offset + 2] = -source[offset + 1];
  }

  const faceStart = model.mesh_faceadr[meshId] * 3;
  const faceCount = model.mesh_facenum[meshId];
  const indices = Uint32Array.from(model.mesh_face.subarray(faceStart, faceStart + faceCount * 3));
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));
  geometry.computeVertexNormals();
  geometry.computeBoundingSphere();
  return geometry;
}

function createFloor(scene) {
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(40, 40),
    new THREE.MeshStandardMaterial({ color: 0x171d22, roughness: 0.88, metalness: 0.05 }),
  );
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = -0.002;
  floor.receiveShadow = true;
  scene.add(floor);

  const grid = new THREE.GridHelper(40, 80, 0x3d4b55, 0x273139);
  grid.material.transparent = true;
  grid.material.opacity = 0.42;
  scene.add(grid);
}

export class HtdThreeRenderer {
  constructor({ container, model, data, pelvisBodyId }) {
    this.container = container;
    this.model = model;
    this.data = data;
    this.pelvisBodyId = pelvisBodyId;
    this.bodyGroups = new Map();
    this.meshGeometryCache = new Map();
    this.lastPelvisPosition = new THREE.Vector3();
    this.currentPelvisPosition = new THREE.Vector3();
    this.renderFrame = () => this.render();

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0c1116);
    this.scene.fog = new THREE.FogExp2(0x0c1116, 0.028);

    this.camera = new THREE.PerspectiveCamera(42, 1, 0.02, 100);
    this.camera.position.set(2.45, 1.45, 2.55);
    this.camera.lookAt(0, 0.72, 0);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.08;
    container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 0.7, 0);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.075;
    this.controls.minDistance = 1.15;
    this.controls.maxDistance = 8;
    this.controls.maxPolarAngle = Math.PI * 0.49;

    this.scene.add(new THREE.HemisphereLight(0xdcefff, 0x202830, 1.6));
    const key = new THREE.DirectionalLight(0xffffff, 3.2);
    key.position.set(-3.5, 6, 3);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.camera.near = 0.2;
    key.shadow.camera.far = 14;
    key.shadow.camera.left = -3;
    key.shadow.camera.right = 3;
    key.shadow.camera.top = 3;
    key.shadow.camera.bottom = -3;
    this.scene.add(key);
    const rim = new THREE.DirectionalLight(0x70d8ff, 1.25);
    rim.position.set(3, 2.5, -4);
    this.scene.add(rim);

    createFloor(this.scene);
    this._createRobotVisuals();
    this.sync(true);

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
    this.resize();
    this.setActive(true);
  }

  _createRobotVisuals() {
    for (let bodyId = 0; bodyId < this.model.nbody; bodyId++) {
      const group = new THREE.Group();
      group.matrixAutoUpdate = true;
      this.bodyGroups.set(bodyId, group);
      this.scene.add(group);
    }

    for (let geomId = 0; geomId < this.model.ngeom; geomId++) {
      // The G1 MJCF puts one render mesh per visual in group 1; group 0 is collision geometry.
      if (this.model.geom_group[geomId] !== 1) continue;
      const meshId = this.model.geom_dataid[geomId];
      if (meshId < 0) continue;
      let geometry = this.meshGeometryCache.get(meshId);
      if (!geometry) {
        geometry = meshGeometry(this.model, meshId);
        this.meshGeometryCache.set(meshId, geometry);
      }
      const [red, green, blue, alpha] = rgbaForGeom(this.model, geomId);
      const material = new THREE.MeshStandardMaterial({
        color: new THREE.Color(red, green, blue),
        roughness: 0.56,
        metalness: 0.18,
        transparent: alpha < 0.999,
        opacity: alpha,
      });
      const visual = new THREE.Mesh(geometry, material);
      visual.castShadow = true;
      visual.receiveShadow = true;
      setMujocoPosition(this.model.geom_pos, geomId, visual.position);
      setMujocoQuaternion(this.model.geom_quat, geomId, visual.quaternion);
      this.bodyGroups.get(this.model.geom_bodyid[geomId]).add(visual);
    }
  }

  sync(first = false) {
    for (const [bodyId, group] of this.bodyGroups) {
      setMujocoPosition(this.data.xpos, bodyId, group.position);
      setMujocoQuaternion(this.data.xquat, bodyId, group.quaternion);
    }

    setMujocoPosition(this.data.xpos, this.pelvisBodyId, this.currentPelvisPosition);
    if (first) {
      this.lastPelvisPosition.copy(this.currentPelvisPosition);
    } else {
      const delta = this.currentPelvisPosition.clone().sub(this.lastPelvisPosition);
      this.camera.position.add(delta);
      this.controls.target.add(delta);
      this.lastPelvisPosition.copy(this.currentPelvisPosition);
    }
  }

  resetCameraTracking() {
    setMujocoPosition(this.data.xpos, this.pelvisBodyId, this.lastPelvisPosition);
  }

  resize() {
    const width = Math.max(1, this.container.clientWidth);
    const height = Math.max(1, this.container.clientHeight);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
  }

  render() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  setActive(active) {
    this.renderer.setAnimationLoop(active ? this.renderFrame : null);
    if (active) {
      this.resize();
      this.render();
    }
  }

  dispose() {
    this.resizeObserver.disconnect();
    this.renderer.setAnimationLoop(null);
    this.controls.dispose();
    const cachedGeometries = new Set(this.meshGeometryCache.values());
    this.scene.traverse((object) => {
      if (object.geometry && !cachedGeometries.has(object.geometry)) object.geometry.dispose?.();
      if (Array.isArray(object.material)) object.material.forEach((material) => material.dispose?.());
      else object.material?.dispose?.();
    });
    for (const geometry of this.meshGeometryCache.values()) geometry.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}
