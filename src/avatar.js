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

function disposeAvatar() {
  clearInterval(blinkTimer);
  if (renderer) {
    renderer.domElement.remove();
    renderer.dispose();
  }
}

export default loadAvatar;
