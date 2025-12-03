// app.js — module
import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.15.0/dist/tf.esm.min.js';

const MODEL_PATH = './model/keypoint_classifier.tflite'; // положи сюди свій файл

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const predEl = document.getElementById('prediction');
const gestureEl = document.getElementById('gesture');
const statusEl = document.getElementById('status');
const logEl = document.getElementById('log');
const startStopBtn = document.getElementById('startStop');

let running = true;
let camera = null;
let tfliteModel = null;
let lastPredLabel = '';
let sameLetterCounter = 0;
const REQUIRED_FRAMES = 150; // як у Python
const FRAME_RATE = 30; // приблизно
const countdownSeconds = Math.floor(REQUIRED_FRAMES / FRAME_RATE);

async function init() {
  statusEl.textContent = 'Ініціалізація tflite...';

  // динамічно завантажимо tfjs-tflite (ESM build)
  // використаємо CDN build зі jsdelivr
  const tfliteModuleUrl = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite/dist/tfjs-tflite.esm.js';
  try {
    const tflite = await import(tfliteModuleUrl);
    // Завантажуємо модель (.tflite) прямо
    tfliteModel = await tflite.loadTFLiteModel(MODEL_PATH);
    statusEl.textContent = 'TFLite модель завантажена.';
    predEl.textContent = 'Модель завантажена — готово';
  } catch (e) {
    console.error('Помилка завантаження tflite або модуля:', e);
    statusEl.textContent = 'Помилка tflite: ' + e.message;
    predEl.textContent = 'Не вдалося завантажити модель. Перевір model/keypoint_classifier.tflite';
    return;
  }

  // Ініціалізація MediaPipe Hands
  const hands = new Hands({locateFile: (file) => {
    return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
  }});
  hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.6,
    minTrackingConfidence: 0.5
  });

  hands.onResults(onResults);

  camera = new Camera(video, {
    onFrame: async () => {
      if (!running) return;
      await hands.send({image: video});
    },
    width: 720,
    height: 540
  });
  camera.start();

  startStopBtn.addEventListener('click', () => {
    running = !running;
    startStopBtn.textContent = running ? 'Stop' : 'Start';
    statusEl.textContent = running ? 'Робота' : 'Зупинено';
  });
}

// Helper: preprocess landmarks similar to Python pre_process_landmark
function preprocessLandmark(landmarkList) {
  // landmarkList — масив з 21 елементів: {x:..., y:...}
  if (!landmarkList || landmarkList.length === 0) return null;
  const temp = landmarkList.map(p => [p.x, p.y]); // вже нормалізовані відносно розміру зображення

  // відняти базову точку (зап'ястя index 0)
  const baseX = temp[0][0], baseY = temp[0][1];
  for (let i = 0; i < temp.length; i++) {
    temp[i][0] = temp[i][0] - baseX;
    temp[i][1] = temp[i][1] - baseY;
  }
  // flatten
  const flat = temp.flat();
  // normalization
  const absVals = flat.map(Math.abs);
  const maxVal = Math.max(...absVals, 1e-6);
  const normalized = flat.map(v => v / maxVal);
  return normalized; // length 42
}

// drawing
function drawLandmarks(landmarks) {
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if (!landmarks) return;
  // use drawing_utils from MediaPipe if available
  try {
    // drawing_utils.drawConnectors etc are global functions included by CDN
    drawConnectors(ctx, landmarks, HAND_CONNECTIONS, {color:'#00FF00', lineWidth:2});
    drawLandmarks(ctx, landmarks, {color:'#FF0000', lineWidth:1});
  } catch (e) {
    // fallback: draw simple circles
    for (const p of landmarks) {
      const x = p.x * canvas.width;
      const y = p.y * canvas.height;
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, 2*Math.PI);
      ctx.fillStyle = 'red';
      ctx.fill();
    }
  }
}

// Predict helper — tries predict(), then run()
async function predictFromTFLite(inputArray) {
  // inputArray — JS array length 42
  if (!tfliteModel) throw new Error('No model loaded');
  // Try using tf.tensor -> predict (some builds accept tf.Tensor)
  try {
    const inputTensor = tf.tensor([inputArray], [1, inputArray.length], 'float32');
    let out = null;
    if (typeof tfliteModel.predict === 'function') {
      out = tfliteModel.predict(inputTensor);
      // out can be tf.Tensor or array
      if (out instanceof tf.Tensor) {
        const data = out.dataSync();
        out.dispose();
        inputTensor.dispose();
        return Array.from(data);
      } else if (Array.isArray(out) || out.buffer) {
        inputTensor.dispose();
        return Array.from(out);
      } else {
        inputTensor.dispose();
        return out;
      }
    } else {
      // fallback: some tflite builds expose "run" that expects TypedArray(s)
      inputTensor.dispose();
    }
  } catch (err) {
    // ignore and try run below
    console.warn('predict() failed, trying run():', err.message);
  }

  // Fallback: try run (some builds)
  try {
    // convert to Float32Array
    const typed = new Float32Array(inputArray);
    // run may accept object {input: typed}
    if (typeof tfliteModel.run === 'function') {
      const res = await tfliteModel.run(typed);
      // res may be TypedArray or object
      if (res && res.length) return Array.from(res);
      if (res && typeof res === 'object') {
        // take first field
        for (const k in res) { return Array.from(res[k]); }
      }
    }
  } catch (err) {
    console.warn('run() failed:', err.message);
  }

  throw new Error('Prediction failed: model API mismatch. See console.');
}

// onResults from MediaPipe
async function onResults(results) {
  // mirror draw: MediaPipe video is mirrored by CSS transform; landmarks are not mirrored — be careful visually.
  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    const lm = results.multiHandLandmarks[0]; // each landmark has x,y,z (normalized)

    // draw
    drawLandmarks(lm);

    // prepare landmarks as [{x,y}, ...] to match preprocess
    const landmarks01 = lm.map(p => ({x: p.x, y: p.y}));

    const input = preprocessLandmark(landmarks01);
    if (input) {
      try {
        const scores = await predictFromTFLite(input); // array of scores
        // pick argmax
        const maxIdx = scores.indexOf(Math.max(...scores));
        predEl.textContent = `Prediction index: ${maxIdx} (score ${scores[maxIdx].toFixed(2)})`;
        // Map index->label: we will try to load labels CSV automatically (if present)
        const label = await getLabelForIndex(maxIdx);
        gestureEl.textContent = `Жест: ${label}`;
        // game logic similar to Python
        if (label === lastPredLabel) {
          sameLetterCounter += 1;
        } else {
          sameLetterCounter = 0;
        }
        lastPredLabel = label;

        const remainingSeconds = Math.max(0, countdownSeconds - Math.floor(sameLetterCounter / FRAME_RATE));
        statusEl.textContent = `frames: ${sameLetterCounter} / ${REQUIRED_FRAMES} · remaining s: ${remainingSeconds}`;

        if (sameLetterCounter >= REQUIRED_FRAMES) {
          logEl.textContent = `Captured letter ${label} (index ${maxIdx})`;
          // here you can call process_letter equivalent (update UI, guessed letters etc.)
          sameLetterCounter = 0;
        }
      } catch (err) {
        console.error('Prediction error:', err);
        statusEl.textContent = 'Prediction error — дивись консоль';
      }
    }
  } else {
    // no hand
    ctx.clearRect(0,0,canvas.width,canvas.height);
    predEl.textContent = 'No hand';
    gestureEl.textContent = '—';
    statusEl.textContent = 'waiting...';
  }
}

// Try to fetch label CSV similar to Python file
let labelList = null;
async function loadLabelsIfExists() {
  try {
    const resp = await fetch('./model/keypoint_classifier_label.csv');
    if (!resp.ok) return null;
    const text = await resp.text();
    labelList = text.split(/\r?\n/).map(s => s.trim()).filter(Boolean);
    console.log('Labels loaded', labelList);
  } catch (e) {
    console.warn('No labels file found', e.message);
  }
}
async function getLabelForIndex(idx) {
  if (!labelList) await loadLabelsIfExists();
  if (labelList && labelList[idx]) {
    // mapping to Ukrainian letters as in Python
    const mapping = {
      "V": "В","Y": "У","R": "Р","A": "А","YA":"Я","N": "Н","I": "І","T": "Т",
      "U": "И","P": "П","G": "Г","E": "Е","Z": "Ж","L": "Л","M": "М","O": "О",
      "C": "С","F": "Ф","SH":"Ш","YU":"Ю","X": "Х","CH":"Ч","B": "Б"
    };
    const lab = labelList[idx];
    return mapping[lab.toUpperCase()] || lab;
  }
  return String(idx);
}

await init();
