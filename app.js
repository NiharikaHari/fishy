import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/+esm";

// ─── DOM Elements ──────────────────────────────────────────────────
const landing = document.getElementById("landing");
const startBtn = document.getElementById("startBtn");
const loadingEl = document.getElementById("loading");
const canvas = document.getElementById("gameCanvas");
const ctx = canvas.getContext("2d");
const video = document.getElementById("camera");
const hintsEl = document.getElementById("hints");
const errorEl = document.getElementById("error");
const errorMsg = document.getElementById("errorMsg");
const noFaceEl = document.getElementById("noFace");

// ─── Constants ─────────────────────────────────────────────────────
const GAZE_SMOOTH = 0.15;
const GAZE_SENSITIVITY = 2.0;
const FISH_SPEED = 0.03;
const FISH_BASE_W = 160;
const FISH_BASE_H = 100;
const MAX_PARTICLES = 100;
const MOUTH_CYCLE_THRESHOLD = 0.15;
const MOUTH_CYCLES_NEEDED = 3;
const MOUTH_WINDOW_MS = 2000;
const MOUTH_COOLDOWN_MS = 500;
const NO_FACE_TIMEOUT_MS = 3000;

// ─── State ─────────────────────────────────────────────────────────
let faceLandmarker = null;
let lastVideoTime = -1;
let running = false;

// Gaze
let gazeX = 0;
let gazeY = 0;
let smoothGazeX = 0;
let smoothGazeY = 0;

// Fish
const fish = {
  x: 0, y: 0,
  targetX: 0, targetY: 0,
  angle: 0,
  scale: 1.0,
  tailPhase: 0,
  tailSpeed: 0.05,
  finPhase: 0,
  mouthOpen: 0,
  leftEyeOpen: 1,
  rightEyeOpen: 1,
  tilt: 0,
  brightness: 0,    // 0 = normal orange, 1 = gold
  puffAmount: 0,     // 0 = normal, 1 = full puffer
  state: "IDLE",
  stateTimer: 0,
  jumpVY: 0,
  jumpOffsetY: 0,
};

// Mouth oscillation detector
const mouthDetector = {
  wasOpen: false,
  cycles: [],      // timestamps of close events
  isActive: false,
  intensity: 0,
  cooldownUntil: 0,
};

// Blendshapes cache
let bs = {};

// Face tracking
let lastFaceTime = 0;
let faceDetected = false;

// Particles
let particles = [];

// Water
let waterTime = 0;

// ─── MediaPipe Setup ───────────────────────────────────────────────
async function initFaceLandmarker() {
  const filesetResolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
  );
  faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: false,
  });
}

async function initCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480, facingMode: "user" },
    audio: false,
  });
  video.srcObject = stream;
  await new Promise(r => { video.onloadeddata = r; });
}

// ─── Detection Loop ────────────────────────────────────────────────
function detect() {
  if (!running) return;

  if (document.hidden) {
    requestAnimationFrame(detect);
    return;
  }

  const now = performance.now();

  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const result = faceLandmarker.detectForVideo(video, now);

    if (result.faceLandmarks && result.faceLandmarks.length > 0) {
      faceDetected = true;
      lastFaceTime = now;
      const landmarks = result.faceLandmarks[0];

      // Parse blendshapes
      bs = {};
      if (result.faceBlendshapes && result.faceBlendshapes.length > 0) {
        for (const b of result.faceBlendshapes[0].categories) {
          bs[b.categoryName] = b.score;
        }
      }

      processGaze(landmarks);
      processMouthOscillation(now);
      processFishState(landmarks, now);
    } else {
      faceDetected = false;
    }
  }

  // No-face warning
  if (!faceDetected && now - lastFaceTime > NO_FACE_TIMEOUT_MS && lastFaceTime > 0) {
    noFaceEl.classList.remove("hidden");
  } else {
    noFaceEl.classList.add("hidden");
  }

  update(now);
  render();
  requestAnimationFrame(detect);
}

// ─── Gaze Estimation ───────────────────────────────────────────────
function processGaze(lm) {
  // Left iris: landmarks 468-472, right iris: 473-477
  const leftIris = avgPoint(lm, [468, 469, 470, 471, 472]);
  const rightIris = avgPoint(lm, [473, 474, 475, 476, 477]);

  // Eye corners: left eye 33(outer)/133(inner), right eye 362(inner)/263(outer)
  const leftOuter = lm[33];
  const leftInner = lm[133];
  const rightInner = lm[362];
  const rightOuter = lm[263];

  // Normalize iris horizontally in eye box
  const leftNormX = (leftIris.x - leftOuter.x) / (leftInner.x - leftOuter.x);
  const rightNormX = (rightIris.x - rightInner.x) / (rightOuter.x - rightInner.x);

  // Vertical: use eye corner midpoint as reference and normalize by eye height
  // Left eye top/bottom landmarks: 159 (top), 145 (bottom)
  const leftEyeMidY = (leftOuter.y + leftInner.y) / 2;
  const leftEyeH = Math.abs(lm[159].y - lm[145].y) || 0.001;
  const leftNormY = (leftIris.y - leftEyeMidY) / leftEyeH;

  // Right eye top/bottom landmarks: 386 (top), 374 (bottom)
  const rightEyeMidY = (rightInner.y + rightOuter.y) / 2;
  const rightEyeH = Math.abs(lm[386].y - lm[374].y) || 0.001;
  const rightNormY = (rightIris.y - rightEyeMidY) / rightEyeH;

  // Average both eyes
  const avgNormX = ((leftNormX + rightNormX) / 2 - 0.5) * GAZE_SENSITIVITY;

  // Vertical: smaller sensitivity so movement stays within screen bounds
  const rawNormY = (leftNormY + rightNormY) / 2;
  const VERT_BIAS = 0.0;
  const VERT_SENSITIVITY = 0.6;
  const avgNormY = (rawNormY - VERT_BIAS) * VERT_SENSITIVITY;

  // Map to screen (invert X for mirror)
  gazeX = (0.5 - avgNormX) * canvas.width;

  // Map vertical to a middle band [0.2, 0.8] of the screen height
  let gyNorm = 0.5 + avgNormY;              // base 0..1-ish
  gyNorm = Math.max(0, Math.min(1, gyNorm));
  const minBand = 0.2;
  const maxBand = 0.8;
  gyNorm = minBand + (maxBand - minBand) * gyNorm;
  gazeY = gyNorm * canvas.height;

  // Smooth
  smoothGazeX += (gazeX - smoothGazeX) * GAZE_SMOOTH;
  smoothGazeY += (gazeY - smoothGazeY) * GAZE_SMOOTH;

  // Clamp
  smoothGazeX = Math.max(0, Math.min(canvas.width, smoothGazeX));
  smoothGazeY = Math.max(0, Math.min(canvas.height, smoothGazeY));

  fish.targetX = smoothGazeX;
  fish.targetY = smoothGazeY;
}

function avgPoint(lm, indices) {
  let x = 0, y = 0;
  for (const i of indices) {
    x += lm[i].x;
    y += lm[i].y;
  }
  return { x: x / indices.length, y: y / indices.length };
}

// ─── Mouth Oscillation Detector ────────────────────────────────────
function processMouthOscillation(now) {
  const jawOpen = bs.jawOpen || 0;
  const isOpen = jawOpen > MOUTH_CYCLE_THRESHOLD;

  // Detect close transition
  if (mouthDetector.wasOpen && !isOpen) {
    mouthDetector.cycles.push(now);
  }
  mouthDetector.wasOpen = isOpen;

  // Prune old cycles
  mouthDetector.cycles = mouthDetector.cycles.filter(t => now - t < MOUTH_WINDOW_MS);

  const count = mouthDetector.cycles.length;

  if (count >= MOUTH_CYCLES_NEEDED && now > mouthDetector.cooldownUntil) {
    mouthDetector.isActive = true;
    mouthDetector.intensity = Math.min(1, count / 5);
    mouthDetector.cooldownUntil = now + MOUTH_COOLDOWN_MS;
  } else if (count < 2) {
    mouthDetector.isActive = false;
    mouthDetector.intensity = Math.max(0, mouthDetector.intensity - 0.02);
  }
}

// ─── Fish State Machine ────────────────────────────────────────────
function processFishState(lm, now) {
  const cheekPuff = (bs.cheekPuff || 0);
  const browUp = ((bs.browOuterUpLeft || 0) + (bs.browOuterUpRight || 0)) / 2;
  const eyeWideL = bs.eyeWideLeft || 0;
  const eyeWideR = bs.eyeWideRight || 0;
  const eyeSquintL = bs.eyeSquintLeft || 0;
  const eyeSquintR = bs.eyeSquintRight || 0;
  const eyeBlinkL = bs.eyeBlinkLeft || 0;
  const eyeBlinkR = bs.eyeBlinkRight || 0;
  const smileL = bs.mouthSmileLeft || 0;
  const smileR = bs.mouthSmileRight || 0;
  const jawOpen = bs.jawOpen || 0;

  // Fish mouth open mirrors user
  fish.mouthOpen += (jawOpen - fish.mouthOpen) * 0.3;

  // Eye open state
  fish.leftEyeOpen += ((1 - eyeBlinkL) - fish.leftEyeOpen) * 0.3;
  fish.rightEyeOpen += ((1 - eyeBlinkR) - fish.rightEyeOpen) * 0.3;

  // Head tilt from eye corner angle
  const leftEyeOuter = lm[33];
  const rightEyeOuter = lm[263];
  const dy = rightEyeOuter.y - leftEyeOuter.y;
  const dx = rightEyeOuter.x - leftEyeOuter.x;
  const headTilt = Math.atan2(dy, dx);
  fish.tilt += (headTilt - fish.tilt) * 0.1;

  // State priority
  if (cheekPuff > 0.4) {
    fish.state = "PUFFED";
  } else if (mouthDetector.isActive) {
    fish.state = "HAPPY";
  } else if (browUp > 0.4 && (eyeWideL + eyeWideR) / 2 > 0.3) {
    if (fish.state !== "SURPRISED") {
      fish.jumpVY = -8;
    }
    fish.state = "SURPRISED";
  } else if ((eyeBlinkL > 0.6 && eyeBlinkR < 0.3) || (eyeBlinkR > 0.6 && eyeBlinkL < 0.3)) {
    fish.state = "WINKING";
  } else if (faceDetected) {
    fish.state = "SWIMMING";
  } else {
    fish.state = "IDLE";
  }

  // Smile overlay — spawn bubbles
  if (smileL + smileR > 0.8) {
    if (Math.random() < 0.3) {
      spawnBubble(fish.x + Math.cos(fish.angle) * 40, fish.y + Math.sin(fish.angle) * 40);
    }
  }
}

// ─── Update ────────────────────────────────────────────────────────
function update(now) {
  const dt = 1; // frame-based

  // Fish movement toward target
  const dx = fish.targetX - fish.x;
  const dy = fish.targetY - fish.y;
  const dist = Math.sqrt(dx * dx + dy * dy);

  if (fish.state !== "IDLE" || dist > 5) {
    const speed = FISH_SPEED * Math.min(dist, 300);
    fish.x += dx * FISH_SPEED;
    fish.y += dy * FISH_SPEED;
  }

  // Face direction of travel
  if (dist > 10) {
    let targetAngle = Math.atan2(dy, dx);
    let angleDiff = targetAngle - fish.angle;
    // Normalize angle diff
    while (angleDiff > Math.PI) angleDiff -= Math.PI * 2;
    while (angleDiff < -Math.PI) angleDiff += Math.PI * 2;
    fish.angle += angleDiff * 0.08;
  }

  // Tail speed based on swim speed
  const swimSpeed = dist > 5 ? Math.min(dist / 100, 1) : 0;
  fish.tailSpeed = 0.05 + swimSpeed * 0.15;
  fish.tailPhase += fish.tailSpeed;
  fish.finPhase += 0.03;

  // State-specific updates
  switch (fish.state) {
    case "PUFFED":
      fish.puffAmount += (1 - fish.puffAmount) * 0.1;
      fish.scale += (1.4 - fish.scale) * 0.1;
      fish.brightness += (0 - fish.brightness) * 0.1;
      break;

    case "HAPPY":
      fish.puffAmount += (0 - fish.puffAmount) * 0.1;
      fish.scale += (1.0 - fish.scale) * 0.1;
      fish.brightness += (1 - fish.brightness) * 0.1;
      // Jiggle
      fish.x += Math.sin(now * 0.02) * 2 * mouthDetector.intensity;
      fish.y += Math.cos(now * 0.025) * 1.5 * mouthDetector.intensity;
      // Sparkles
      if (Math.random() < 0.2) {
        spawnSparkle(
          fish.x + (Math.random() - 0.5) * 100,
          fish.y + (Math.random() - 0.5) * 60
        );
      }
      break;

    case "SURPRISED":
      fish.puffAmount += (0 - fish.puffAmount) * 0.1;
      fish.scale += (1.15 - fish.scale) * 0.1;
      fish.brightness += (0 - fish.brightness) * 0.1;
      // Jump
      fish.jumpVY += 0.3; // gravity
      fish.jumpOffsetY += fish.jumpVY;
      if (fish.jumpOffsetY > 0) {
        fish.jumpOffsetY = 0;
        fish.jumpVY = 0;
      }
      // Particle burst on entry
      if (fish.stateTimer < 5) {
        for (let i = 0; i < 3; i++) {
          spawnSparkle(
            fish.x + (Math.random() - 0.5) * 80,
            fish.y + fish.jumpOffsetY + (Math.random() - 0.5) * 60
          );
        }
      }
      fish.stateTimer++;
      break;

    case "WINKING":
      fish.puffAmount += (0 - fish.puffAmount) * 0.1;
      fish.scale += (1.0 - fish.scale) * 0.1;
      fish.brightness += (0 - fish.brightness) * 0.1;
      break;

    case "SWIMMING":
      fish.puffAmount += (0 - fish.puffAmount) * 0.1;
      fish.scale += (1.0 - fish.scale) * 0.1;
      fish.brightness += (0 - fish.brightness) * 0.1;
      fish.jumpOffsetY += (0 - fish.jumpOffsetY) * 0.1;
      break;

    case "IDLE":
    default:
      fish.puffAmount += (0 - fish.puffAmount) * 0.1;
      fish.scale += (1.0 - fish.scale) * 0.1;
      fish.brightness += (0 - fish.brightness) * 0.1;
      fish.jumpOffsetY += (0 - fish.jumpOffsetY) * 0.1;
      // Sinusoidal bob
      fish.y += Math.sin(now * 0.002) * 0.5;
      break;
  }

  if (fish.state !== "SURPRISED") {
    fish.stateTimer = 0;
  }

  // Update particles
  updateParticles();

  // Ambient bubbles
  if (Math.random() < 0.02) {
    spawnBubble(
      Math.random() * canvas.width,
      canvas.height + 10
    );
  }

  waterTime += 0.01;
}

// ─── Particles ─────────────────────────────────────────────────────
function spawnBubble(x, y) {
  if (particles.length >= MAX_PARTICLES) return;
  particles.push({
    type: "bubble",
    x, y,
    vx: (Math.random() - 0.5) * 0.5,
    vy: -1 - Math.random() * 1.5,
    r: 3 + Math.random() * 6,
    life: 1,
    decay: 0.003 + Math.random() * 0.003,
    wobblePhase: Math.random() * Math.PI * 2,
    wobbleSpeed: 0.02 + Math.random() * 0.02,
  });
}

function spawnSparkle(x, y) {
  if (particles.length >= MAX_PARTICLES) return;
  particles.push({
    type: "sparkle",
    x, y,
    vx: (Math.random() - 0.5) * 2,
    vy: (Math.random() - 0.5) * 2,
    size: 4 + Math.random() * 6,
    life: 1,
    decay: 0.02 + Math.random() * 0.02,
    rotation: Math.random() * Math.PI,
    rotSpeed: 0.05 + Math.random() * 0.05,
  });
}

function updateParticles() {
  for (let i = particles.length - 1; i >= 0; i--) {
    const p = particles[i];
    p.life -= p.decay;
    if (p.life <= 0) {
      particles.splice(i, 1);
      continue;
    }

    if (p.type === "bubble") {
      p.wobblePhase += p.wobbleSpeed;
      p.x += p.vx + Math.sin(p.wobblePhase) * 0.5;
      p.y += p.vy;
    } else {
      p.x += p.vx;
      p.y += p.vy;
      p.vx *= 0.98;
      p.vy *= 0.98;
      p.rotation += p.rotSpeed;
      p.size *= 0.99;
    }
  }
}

// ─── Rendering ─────────────────────────────────────────────────────
function render() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;

  drawWaterBackground();
  drawParticles();
  drawFish();
  drawGazeDebug();
}

// ─── Gaze Debug Marker ─────────────────────────────────────────────
function drawGazeDebug() {
  if (!faceDetected) return;
  const x = smoothGazeX;
  const y = smoothGazeY;

  // Crosshair
  ctx.save();
  ctx.strokeStyle = "rgba(255, 80, 80, 0.7)";
  ctx.lineWidth = 2;

  // Horizontal line
  ctx.beginPath();
  ctx.moveTo(x - 15, y);
  ctx.lineTo(x + 15, y);
  ctx.stroke();

  // Vertical line
  ctx.beginPath();
  ctx.moveTo(x, y - 15);
  ctx.lineTo(x, y + 15);
  ctx.stroke();

  // Circle
  ctx.beginPath();
  ctx.arc(x, y, 10, 0, Math.PI * 2);
  ctx.stroke();

  // Dot
  ctx.beginPath();
  ctx.arc(x, y, 3, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(255, 80, 80, 0.9)";
  ctx.fill();

  // Label
  ctx.font = "12px monospace";
  ctx.fillStyle = "rgba(255, 80, 80, 0.8)";
  ctx.fillText(`gaze (${Math.round(x)}, ${Math.round(y)})`, x + 16, y - 8);
  ctx.restore();
}

// ─── Water Background ──────────────────────────────────────────────
function drawWaterBackground() {
  // Gradient
  const grad = ctx.createLinearGradient(0, 0, 0, canvas.height);
  grad.addColorStop(0, "#0a1628");
  grad.addColorStop(0.3, "#0f2847");
  grad.addColorStop(0.6, "#0f2847");
  grad.addColorStop(1, "#0a1628");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Light rays
  ctx.save();
  ctx.globalAlpha = 0.03;
  for (let i = 0; i < 4; i++) {
    const x = canvas.width * (0.2 + i * 0.2) + Math.sin(waterTime + i) * 30;
    const w = 60 + Math.sin(waterTime * 0.5 + i * 2) * 20;
    ctx.beginPath();
    ctx.moveTo(x - w / 2, 0);
    ctx.lineTo(x + w / 2, 0);
    ctx.lineTo(x + w * 1.5, canvas.height);
    ctx.lineTo(x - w * 1.5, canvas.height);
    ctx.closePath();
    ctx.fillStyle = "#4a9eff";
    ctx.fill();
  }
  ctx.restore();

  // Sine wave lines
  ctx.save();
  ctx.globalAlpha = 0.08;
  ctx.strokeStyle = "#4a9eff";
  ctx.lineWidth = 1;
  for (let i = 0; i < 5; i++) {
    const baseY = canvas.height * (0.2 + i * 0.15);
    ctx.beginPath();
    for (let x = 0; x <= canvas.width; x += 10) {
      const y = baseY + Math.sin(x * 0.005 + waterTime * 2 + i * 1.5) * 15;
      if (x === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
  ctx.restore();
}

// ─── Draw Fish ─────────────────────────────────────────────────────
function drawFish() {
  const fx = fish.x;
  const fy = fish.y + fish.jumpOffsetY;
  const s = fish.scale;
  const w = FISH_BASE_W * s;
  const h = FISH_BASE_H * s;

  ctx.save();
  ctx.translate(fx, fy);
  ctx.rotate(fish.angle + fish.tilt);
  ctx.scale(s, s);

  // Puff expansion
  const puff = fish.puffAmount;
  const bodyScaleX = 1 + puff * 0.4;
  const bodyScaleY = 1 + puff * 0.4;

  // ── Tail ──
  const tailSway = Math.sin(fish.tailPhase) * 20;
  ctx.save();
  ctx.translate(-50, 0);
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.quadraticCurveTo(-25 + tailSway, -25, -50 + tailSway * 1.3, -20);
  ctx.lineTo(-50 + tailSway * 1.3, 20);
  ctx.quadraticCurveTo(-25 + tailSway, 25, 0, 0);
  ctx.closePath();
  const tailGrad = ctx.createLinearGradient(0, 0, -50, 0);
  tailGrad.addColorStop(0, lerpColor("#e8751a", "#ffc107", fish.brightness));
  tailGrad.addColorStop(1, lerpColor("#d4520a", "#ff9800", fish.brightness));
  ctx.fillStyle = tailGrad;
  ctx.fill();
  ctx.restore();

  // ── Dorsal Fin ──
  ctx.save();
  ctx.scale(bodyScaleX, bodyScaleY);
  const dorsalWave = Math.sin(fish.finPhase) * 5;
  ctx.beginPath();
  ctx.moveTo(-10, -35);
  ctx.quadraticCurveTo(10, -65 + dorsalWave, 30, -35);
  ctx.closePath();
  ctx.fillStyle = lerpColor("#d4520a", "#ff9800", fish.brightness);
  ctx.fill();
  ctx.restore();

  // ── Pectoral Fin (bottom) ──
  ctx.save();
  ctx.scale(bodyScaleX, bodyScaleY);
  const pectoralWave = Math.sin(fish.finPhase * 1.3 + 1) * 8;
  ctx.beginPath();
  ctx.moveTo(-5, 25);
  ctx.quadraticCurveTo(-20, 45 + pectoralWave, -5, 50 + pectoralWave);
  ctx.quadraticCurveTo(5, 40 + pectoralWave, 5, 25);
  ctx.closePath();
  ctx.fillStyle = lerpColor("#d4520a", "#ff9800", fish.brightness);
  ctx.fill();
  ctx.restore();

  // ── Body ──
  ctx.save();
  ctx.scale(bodyScaleX, bodyScaleY);
  ctx.beginPath();
  ctx.ellipse(0, 0, 60, 35, 0, 0, Math.PI * 2);
  const bodyGrad = ctx.createRadialGradient(-10, -10, 5, 0, 0, 60);
  bodyGrad.addColorStop(0, lerpColor("#ffb347", "#ffe082", fish.brightness));
  bodyGrad.addColorStop(0.6, lerpColor("#f8a634", "#ffc107", fish.brightness));
  bodyGrad.addColorStop(1, lerpColor("#e8751a", "#ff9800", fish.brightness));
  ctx.fillStyle = bodyGrad;
  ctx.fill();

  // ── Scales ──
  ctx.save();
  ctx.clip();
  ctx.globalAlpha = 0.15;
  ctx.strokeStyle = lerpColor("#d4520a", "#ff9800", fish.brightness);
  ctx.lineWidth = 0.8;
  for (let row = -2; row <= 2; row++) {
    for (let col = -3; col <= 2; col++) {
      const sx = col * 18 + (row % 2) * 9;
      const sy = row * 14;
      ctx.beginPath();
      ctx.arc(sx, sy, 9, -0.5, Math.PI + 0.5, false);
      ctx.stroke();
    }
  }
  ctx.restore();
  ctx.restore();

  // ── Puffer Spikes ──
  if (puff > 0.1) {
    ctx.save();
    ctx.globalAlpha = puff;
    ctx.fillStyle = lerpColor("#e8751a", "#ff9800", fish.brightness);
    const spikeCount = 16;
    for (let i = 0; i < spikeCount; i++) {
      const ang = (i / spikeCount) * Math.PI * 2;
      const baseR = 38 * bodyScaleX;
      const spikeLen = 15 * puff;
      const bx = Math.cos(ang) * baseR;
      const by = Math.sin(ang) * baseR * (35 / 60);
      const ex = Math.cos(ang) * (baseR + spikeLen);
      const ey = Math.sin(ang) * (baseR + spikeLen) * (35 / 60);
      const perpX = -Math.sin(ang) * 4;
      const perpY = Math.cos(ang) * 4 * (35 / 60);
      ctx.beginPath();
      ctx.moveTo(bx + perpX, by + perpY);
      ctx.lineTo(ex, ey);
      ctx.lineTo(bx - perpX, by - perpY);
      ctx.closePath();
      ctx.fill();
    }
    ctx.restore();
  }

  // ── Eyes ──
  drawEye(22, -12, fish.rightEyeOpen, fish.state === "SURPRISED"); // right eye (facing right)
  drawEye(22, 12, fish.leftEyeOpen, fish.state === "SURPRISED");   // left eye

  // ── Mouth ──
  const mouthX = 48 * bodyScaleX;
  const mouthOpen = fish.mouthOpen;
  if (mouthOpen > 0.05) {
    ctx.beginPath();
    ctx.ellipse(mouthX, 3, 6 + mouthOpen * 6, 3 + mouthOpen * 8, 0, 0, Math.PI * 2);
    ctx.fillStyle = "#c62828";
    ctx.fill();
  } else {
    ctx.beginPath();
    ctx.arc(mouthX, 5, 10, 0.1, 0.6);
    ctx.strokeStyle = "#c62828";
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  ctx.restore();
}

function drawEye(ox, oy, openAmount, wide) {
  const eyeR = wide ? 14 : 10;
  const pupilR = wide ? 6 : 5;

  // White
  ctx.beginPath();
  ctx.arc(ox, oy, eyeR, 0, Math.PI * 2);
  ctx.fillStyle = "#fff";
  ctx.fill();

  // Pupil
  ctx.beginPath();
  ctx.arc(ox + 2, oy, pupilR, 0, Math.PI * 2);
  ctx.fillStyle = "#1a1a2e";
  ctx.fill();

  // Highlight
  ctx.beginPath();
  ctx.arc(ox + 4, oy - 3, 2.5, 0, Math.PI * 2);
  ctx.fillStyle = "#fff";
  ctx.fill();

  // Eyelid (closes from top)
  if (openAmount < 0.95) {
    const lidClose = 1 - openAmount;
    ctx.save();
    ctx.beginPath();
    ctx.rect(ox - eyeR - 1, oy - eyeR - 1, eyeR * 2 + 2, eyeR * 2 * lidClose + 1);
    ctx.clip();
    ctx.beginPath();
    ctx.arc(ox, oy, eyeR + 1, 0, Math.PI * 2);
    ctx.fillStyle = lerpColor("#f8a634", "#ffc107", fish.brightness);
    ctx.fill();
    ctx.restore();
  }
}

// ─── Draw Particles ────────────────────────────────────────────────
function drawParticles() {
  for (const p of particles) {
    ctx.save();

    if (p.type === "bubble") {
      ctx.globalAlpha = p.life * 0.5;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(180, 220, 255, 0.6)";
      ctx.lineWidth = 1;
      ctx.stroke();
      // Highlight
      ctx.beginPath();
      ctx.arc(p.x - p.r * 0.3, p.y - p.r * 0.3, p.r * 0.3, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(255, 255, 255, ${p.life * 0.4})`;
      ctx.fill();
    } else if (p.type === "sparkle") {
      ctx.globalAlpha = p.life;
      ctx.translate(p.x, p.y);
      ctx.rotate(p.rotation);
      ctx.fillStyle = "#ffd700";
      drawStar(0, 0, p.size);
    }

    ctx.restore();
  }
}

function drawStar(x, y, size) {
  const s = size / 2;
  ctx.beginPath();
  // 4-pointed star
  ctx.moveTo(x, y - s);
  ctx.quadraticCurveTo(x + s * 0.15, y - s * 0.15, x + s, y);
  ctx.quadraticCurveTo(x + s * 0.15, y + s * 0.15, x, y + s);
  ctx.quadraticCurveTo(x - s * 0.15, y + s * 0.15, x - s, y);
  ctx.quadraticCurveTo(x - s * 0.15, y - s * 0.15, x, y - s);
  ctx.closePath();
  ctx.fill();
}

// ─── Helpers ───────────────────────────────────────────────────────
function lerpColor(colorA, colorB, t) {
  // Parse hex
  const a = hexToRgb(colorA);
  const b = hexToRgb(colorB);
  const r = Math.round(a.r + (b.r - a.r) * t);
  const g = Math.round(a.g + (b.g - a.g) * t);
  const bl = Math.round(a.b + (b.b - a.b) * t);
  return `rgb(${r},${g},${bl})`;
}

function hexToRgb(hex) {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return {
    r: parseInt(result[1], 16),
    g: parseInt(result[2], 16),
    b: parseInt(result[3], 16),
  };
}

// ─── Resize ────────────────────────────────────────────────────────
function resize() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  // Initialize fish position to center if not set
  if (fish.x === 0 && fish.y === 0) {
    fish.x = canvas.width / 2;
    fish.y = canvas.height / 2;
    fish.targetX = fish.x;
    fish.targetY = fish.y;
    smoothGazeX = fish.x;
    smoothGazeY = fish.y;
  }
}

window.addEventListener("resize", resize);

// ─── UX Flow ───────────────────────────────────────────────────────
startBtn.addEventListener("click", async () => {
  landing.classList.add("hidden");
  loadingEl.classList.remove("hidden");

  try {
    await Promise.all([initFaceLandmarker(), initCamera()]);

    loadingEl.classList.add("hidden");
    canvas.classList.add("visible");
    video.classList.add("visible");

    resize();
    running = true;
    lastFaceTime = performance.now();

    // Show hints (persistent)
    hintsEl.classList.remove("hidden");

    detect();
  } catch (err) {
    console.error(err);
    loadingEl.classList.add("hidden");
    if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
      errorMsg.textContent = "Camera access denied. Please allow camera access and refresh the page to play.";
    } else {
      errorMsg.textContent = `Something went wrong: ${err.message}. Please refresh and try again.`;
    }
    errorEl.classList.remove("hidden");
  }
});
