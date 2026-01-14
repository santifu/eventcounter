import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0';

// Configure
env.allowLocalModels = false;

// State
let detector = null;
let crowdSession = null;
let currentMode = 'detection';
let modelsLoaded = { detection: false, crowd: false };

// ONNX model URL for crowd counting (CSRNet from muasifk)
const CROWD_MODEL_URL = 'https://huggingface.co/muasifk/CSRNet/resolve/main/model1_A.onnx';

// Animal labels from COCO dataset
const ANIMAL_LABELS = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'];

document.addEventListener('DOMContentLoaded', init);

async function init() {
    setupModeButtons();
    setupUploader();
    await loadModels();
}

function setupModeButtons() {
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentMode = btn.dataset.mode;
            resetResults();
        });
    });
}

function setupUploader() {
    const uploader = document.getElementById('uploader');
    uploader.addEventListener('change', handleImageUpload);
}

async function loadModels() {
    updateStatus('Cargando modelos de IA...', true);
    showProgress(true);

    try {
        // Load DETR for detection mode
        updateStatus('Cargando modelo de detección (DETR)...', true);
        detector = await pipeline('object-detection', 'Xenova/detr-resnet-50', {
            progress_callback: (progress) => {
                if (progress.status === 'progress') {
                    const pct = Math.round((progress.loaded / progress.total) * 100);
                    setProgress(pct / 2); // First half
                }
            }
        });
        modelsLoaded.detection = true;
        console.log('DETR cargado.');

        // Load ONNX crowd model
        updateStatus('Cargando modelo de multitudes...', true);
        setProgress(60);

        crowdSession = await ort.InferenceSession.create(CROWD_MODEL_URL, {
            executionProviders: ['wasm']
        });
        modelsLoaded.crowd = true;
        console.log('Modelo de multitudes cargado.');

        setProgress(100);
        showProgress(false);
        updateStatus('', false);
        enableUpload();

    } catch (err) {
        console.error('Error cargando modelos:', err);
        updateStatus('Error: ' + err.message, true);
        showProgress(false);

        // Enable upload if at least detection works
        if (modelsLoaded.detection) {
            enableUpload();
            updateStatus('Modo multitud no disponible. Detección activa.', true);
        }
    }
}

function enableUpload() {
    const label = document.getElementById('upload-label');
    label.classList.remove('disabled');
    label.textContent = '📷 Seleccionar Imagen';
}

async function handleImageUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    updateStatus('Analizando imagen...', true);
    resetResults();

    const img = new Image();
    img.onload = async () => {
        if (currentMode === 'detection') {
            await analyzeDetection(img);
        } else {
            await analyzeCrowd(img);
        }
    };
    img.src = URL.createObjectURL(file);
}

// === DETECTION MODE ===
async function analyzeDetection(img) {
    if (!detector) {
        updateStatus('Modelo de detección no disponible', true);
        return;
    }

    try {
        const results = await detector(img.src, { threshold: 0.7 });

        const people = results.filter(r => r.label === 'person');
        const animals = results.filter(r => ANIMAL_LABELS.includes(r.label));
        const count = people.length + animals.length;

        showResult(people.length, 'Personas detectadas');
        drawDetectionResults(img, results);
        updateStatus('', false);

    } catch (err) {
        console.error('Error en detección:', err);
        updateStatus('Error: ' + err.message, true);
    }
}

function drawDetectionResults(img, results) {
    const container = document.getElementById('canvas-container');
    container.innerHTML = '';

    const canvas = document.createElement('canvas');
    canvas.width = img.width;
    canvas.height = img.height;
    container.appendChild(canvas);

    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);

    results.forEach(({ label, score, box }) => {
        const isPerson = label === 'person';
        const isAnimal = ANIMAL_LABELS.includes(label);
        if (!isPerson && !isAnimal) return;

        const { xmin, ymin, xmax, ymax } = box;
        const color = isPerson ? '#00c853' : '#ff9800';

        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);

        const labelText = `${label} ${Math.round(score * 100)}%`;
        ctx.font = 'bold 14px Arial';
        const textWidth = ctx.measureText(labelText).width;

        ctx.fillStyle = color;
        ctx.fillRect(xmin, ymin - 22, textWidth + 10, 22);

        ctx.fillStyle = 'white';
        ctx.fillText(labelText, xmin + 5, ymin - 6);
    });
}

// === CROWD MODE ===
async function analyzeCrowd(img) {
    if (!crowdSession) {
        updateStatus('Modelo de multitudes no disponible', true);
        return;
    }

    try {
        updateStatus('Procesando mapa de densidad...', true);

        // Prepare image for ONNX (CSRNet expects 1024x768)
        const { tensor, originalWidth, originalHeight } = await prepareImageForONNX(img);

        // Run inference
        const feeds = { input: tensor };
        const output = await crowdSession.run(feeds);

        // Get density map
        const densityMap = output[Object.keys(output)[0]];
        const densityData = densityMap.data;

        // Calculate total count (sum of density map)
        let totalCount = 0;
        for (let i = 0; i < densityData.length; i++) {
            totalCount += densityData[i];
        }
        totalCount = Math.round(totalCount);

        showResult(totalCount, 'Personas estimadas (densidad)');
        drawDensityHeatmap(img, densityData, densityMap.dims);
        updateStatus('', false);

    } catch (err) {
        console.error('Error en conteo de multitudes:', err);
        updateStatus('Error: ' + err.message, true);
    }
}

async function prepareImageForONNX(img) {
    // CSRNet expects 1024x768 input
    const targetWidth = 1024;
    const targetHeight = 768;

    const canvas = document.createElement('canvas');
    canvas.width = targetWidth;
    canvas.height = targetHeight;
    const ctx = canvas.getContext('2d');

    // Resize image to fill canvas
    ctx.drawImage(img, 0, 0, targetWidth, targetHeight);

    const imageData = ctx.getImageData(0, 0, targetWidth, targetHeight);
    const { data } = imageData;

    // Convert to RGB float tensor (CHW format), normalize by /255 only
    const floatData = new Float32Array(3 * targetWidth * targetHeight);

    for (let i = 0; i < targetWidth * targetHeight; i++) {
        floatData[i] = data[i * 4] / 255;                              // R
        floatData[targetWidth * targetHeight + i] = data[i * 4 + 1] / 255;      // G
        floatData[2 * targetWidth * targetHeight + i] = data[i * 4 + 2] / 255;  // B
    }

    const tensor = new ort.Tensor('float32', floatData, [1, 3, targetHeight, targetWidth]);

    return { tensor, originalWidth: img.width, originalHeight: img.height };
}

function drawDensityHeatmap(img, densityData, dims) {
    const container = document.getElementById('canvas-container');
    container.innerHTML = '';

    const canvas = document.createElement('canvas');
    canvas.width = img.width;
    canvas.height = img.height;
    container.appendChild(canvas);

    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);

    // Create heatmap overlay
    const heatmapCanvas = document.createElement('canvas');
    const heatmapWidth = dims[3] || dims[2] || 64;
    const heatmapHeight = dims[2] || dims[1] || 64;
    heatmapCanvas.width = heatmapWidth;
    heatmapCanvas.height = heatmapHeight;
    const heatmapCtx = heatmapCanvas.getContext('2d');

    // Find max value for normalization
    let maxVal = 0;
    for (let i = 0; i < densityData.length; i++) {
        if (densityData[i] > maxVal) maxVal = densityData[i];
    }

    // Draw heatmap
    const heatmapImageData = heatmapCtx.createImageData(heatmapWidth, heatmapHeight);
    for (let i = 0; i < densityData.length; i++) {
        const normalized = maxVal > 0 ? densityData[i] / maxVal : 0;
        const color = getHeatmapColor(normalized);
        heatmapImageData.data[i * 4] = color.r;
        heatmapImageData.data[i * 4 + 1] = color.g;
        heatmapImageData.data[i * 4 + 2] = color.b;
        heatmapImageData.data[i * 4 + 3] = Math.floor(normalized * 180); // Semi-transparent
    }
    heatmapCtx.putImageData(heatmapImageData, 0, 0);

    // Overlay heatmap on original image
    ctx.drawImage(heatmapCanvas, 0, 0, img.width, img.height);
}

function getHeatmapColor(value) {
    // Blue -> Cyan -> Green -> Yellow -> Red
    const colors = [
        { r: 0, g: 0, b: 255 },    // Blue (low)
        { r: 0, g: 255, b: 255 },  // Cyan
        { r: 0, g: 255, b: 0 },    // Green
        { r: 255, g: 255, b: 0 },  // Yellow
        { r: 255, g: 0, b: 0 }     // Red (high)
    ];

    const idx = value * (colors.length - 1);
    const lower = Math.floor(idx);
    const upper = Math.ceil(idx);
    const t = idx - lower;

    if (lower === upper) return colors[lower];

    return {
        r: Math.round(colors[lower].r + t * (colors[upper].r - colors[lower].r)),
        g: Math.round(colors[lower].g + t * (colors[upper].g - colors[lower].g)),
        b: Math.round(colors[lower].b + t * (colors[upper].b - colors[lower].b))
    };
}

// === UI HELPERS ===
function showResult(count, label) {
    document.getElementById('result-box').style.display = 'block';
    document.getElementById('result-count').textContent = count;
    document.getElementById('result-label').textContent = label;
}

function resetResults() {
    document.getElementById('result-box').style.display = 'none';
    document.getElementById('canvas-container').innerHTML = '';
}

function updateStatus(msg, show) {
    const el = document.getElementById('status-message');
    if (el) {
        el.textContent = msg;
        el.style.display = show ? 'block' : 'none';
    }
}

function showProgress(show) {
    const el = document.getElementById('progress-bar');
    if (el) el.style.display = show ? 'block' : 'none';
}

function setProgress(pct) {
    const el = document.getElementById('progress-fill');
    if (el) el.style.width = pct + '%';
}