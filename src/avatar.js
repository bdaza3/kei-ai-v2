import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';

let renderer, scene, camera, controls, vrmInstance, blinkTimer;

export async function loadAvatar(container, modelUrl) {
  const containerEl = typeof container === 'string' ? document.getElementById(container) : container;
  if (!containerEl) throw new Error('Avatar container not found');

  scene = new THREE.Scene();

  camera = new THREE.PerspectiveCamera(45, containerEl.clientWidth / containerEl.clientHeight, 0.1, 100);
  camera.position.set(0.0, 1.4, 2.0);

  renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(containerEl.clientWidth, containerEl.clientHeight);
  renderer.outputEncoding = THREE.sRGBEncoding;
  containerEl.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 1.4, 0);
  controls.update();

  const dir = new THREE.DirectionalLight(0xffffff, 1);
  dir.position.set(1, 1, 1);
  scene.add(dir);
  scene.add(new THREE.AmbientLight(0xffffff, 0.6));

  const loader = new GLTFLoader();
  loader.crossOrigin = 'anonymous';
  loader.register((parser) => new VRMLoaderPlugin(parser));
  const gltf = await loader.loadAsync(modelUrl);
  const vrm = gltf.userData.vrm;
  if (!vrm) throw new Error('VRM not found in gltf.userData.vrm');

  // performance helpers
  try { VRMUtils.removeUnnecessaryVertices(gltf.scene); } catch (e) {}
  try { VRMUtils.combineSkeletons(gltf.scene); } catch (e) {}
  try { VRMUtils.combineMorphs(vrm); } catch (e) {}

  vrm.scene.rotation.y = Math.PI;
  scene.add(vrm.scene);
  vrmInstance = vrm;

  if (vrmInstance.lookAt) vrmInstance.lookAt.target = camera;

  startLoop();
  startBlinking();

  window.addEventListener('resize', onWindowResize);

  return {
    vrm: vrmInstance,
    setEmotion: setEmotion,
    setMouth: setMouth,
    tiltHead: tiltHead,
    nod: nod,
    waveArm: waveArm,
    dispose: disposeAvatar
  };
}

function onWindowResize() {
  if (!camera || !renderer) return;
  const parent = renderer.domElement.parentElement;
  camera.aspect = parent.clientWidth / parent.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(parent.clientWidth, parent.clientHeight);
}

function startLoop() {
  const clock = new THREE.Clock();
  clock.start();
  (function animate() {
    requestAnimationFrame(animate);
    const delta = clock.getDelta();
    const t = clock.elapsedTime;
    if (vrmInstance) {
      vrmInstance.scene.position.y = Math.sin(t * 0.5) * 0.01;
      vrmInstance.scene.rotation.y += Math.sin(t * 0.3) * 0.0008;
      try { vrmInstance.update(delta); } catch (e) {}
    }
    renderer.render(scene, camera);
  })();
}

function startBlinking() {
  clearInterval(blinkTimer);
  blinkTimer = setInterval(() => {
    if (!vrmInstance) return;
    const proxy = vrmInstance.expressionManager;
    if (proxy && proxy.setValue) {
      try {
        proxy.setValue('blinkLeft', 1.0);
        proxy.setValue('blinkRight', 1.0);
        setTimeout(() => {
          proxy.setValue('blinkLeft', 0.0);
          proxy.setValue('blinkRight', 0.0);
        }, 120);
      } catch (e) {}
    }
  }, 3000 + Math.random() * 2000);
}

function setEmotion(name, value = 1.0) {
  if (!vrmInstance) return;
  const proxy = vrmInstance.expressionManager;
  if (!proxy || !proxy.setValue) return;
  const groups = ['neutral', 'happy', 'sad', 'angry', 'surprised', 'relaxed', 'joy'];
  groups.forEach(g => { try { proxy.setValue(g, 0); } catch (e) {} });
  try { proxy.setValue(name, value); } catch (e) {}
}

function setMouth(value) {
  if (!vrmInstance) return;
  const proxy = vrmInstance.expressionManager;
  if (!proxy || !proxy.setValue) return;
  const v = Math.max(0, Math.min(1, value));
  const candidates = ['aa', 'a', 'mouthOpen', 'vowel', 'O', 'I'];
  candidates.forEach(c => { try { proxy.setValue(c, v); } catch (e) {} });
}

// --- Bone animation helpers ---
function findBone(...keys) {
  if (!vrmInstance) return null;
  try {
    if (vrmInstance.humanoid && typeof vrmInstance.humanoid.getBoneNode === 'function') {
      for (const k of keys) {
        try {
          const node = vrmInstance.humanoid.getBoneNode(k);
          if (node) return node;
        } catch (e) {}
      }
    }
  } catch (e) {}
  // Fallback: search by object name on the scene
  for (const k of keys) {
    const n = vrmInstance.scene.getObjectByName(k) || vrmInstance.scene.getObjectByName(k[0].toUpperCase() + k.slice(1));
    if (n) return n;
  }
  let found = null;
  vrmInstance.scene.traverse(n => { if (!found && n.name && keys.includes(n.name.toLowerCase())) found = n; });
  return found;
}

function animateBoneQuaternion(bone, targetQuat, durationMs = 300) {
  return new Promise(resolve => {
    const start = bone.quaternion.clone();
    const t0 = performance.now();
    function tick() {
      const t = Math.min(1, (performance.now() - t0) / durationMs);
      THREE.Quaternion.slerp(start, targetQuat, bone.quaternion, t);
      bone.updateMatrixWorld(true);
      if (t < 1) requestAnimationFrame(tick); else resolve();
    }
    tick();
  });
}

function rotateBoneBy(bone, axis, angleRad, durationMs = 300) {
  const start = bone.quaternion.clone();
  const q = new THREE.Quaternion().setFromAxisAngle(axis.clone().normalize(), angleRad);
  const target = start.clone().multiply(q);
  return animateBoneQuaternion(bone, target, durationMs);
}

async function tiltHead(deg = 10, duration = 300) {
  if (!vrmInstance) return;
  const rad = THREE.MathUtils.degToRad(deg);
  const bone = findBone('neck', 'head');
  if (!bone) return;
  // roll (tilt) around local Z
  await rotateBoneBy(bone, new THREE.Vector3(0, 0, 1), rad, duration);
}

async function nod(times = 1, amplitude = 15, speed = 180) {
  if (!vrmInstance) return;
  const bone = findBone('neck', 'head');
  if (!bone) return;
  const original = bone.quaternion.clone();
  const rad = THREE.MathUtils.degToRad(amplitude);
  for (let i = 0; i < times; i++) {
    await rotateBoneBy(bone, new THREE.Vector3(1, 0, 0), rad, speed);
    await animateBoneQuaternion(bone, original.clone(), speed);
  }
}

async function waveArm(side = 'left', deg = 40, duration = 600) {
  if (!vrmInstance) return;
  const key = side === 'left' ? 'leftUpperArm' : 'rightUpperArm';
  const arm = findBone(key);
  if (!arm) return;
  const original = arm.quaternion.clone();
  const rad = THREE.MathUtils.degToRad(deg);
  const q = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), rad);
  const target = original.clone().multiply(q);
  await animateBoneQuaternion(arm, target, duration / 2);
  await animateBoneQuaternion(arm, original.clone(), duration / 2);
  await animateBoneQuaternion(arm, target, duration / 2);
  await animateBoneQuaternion(arm, original.clone(), duration / 2);
}

function disposeAvatar() {
  clearInterval(blinkTimer);
  if (renderer) {
    renderer.domElement.remove();
    renderer.dispose();
  }
}

export default loadAvatar;
