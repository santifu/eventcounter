import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0';

env.allowLocalModels = false;

// State
let detector = null;
let crowdSession = null;
let modelsReady = { detection: false, crowd: false };

const CROWD_MODEL_URL = 'https://huggingface.co/muasifk/CSRNet/resolve/main/model1_A.onnx';

document.addEventListener('DOMContentLoaded', init);

async function init() {
    setupUploader();
    await loadModels();
}

function setupUploader() {
    document.getElementById('uploader').addEventListener('change', handleImageUpload);
}

// === MODEL LOADING ===

async function loadModels() {
    showModelProgress(true);
    setProgressText('Descargando DETR...');

    try {
        detector = await pipeline('object-detection', 'Xenova/detr-resnet-50', {
            progress_callback: (p) => {
                if (p.status === 'progress') {
                    const pct = Math.round((p.loaded / p.total) * 60);
                    setProgress(pct);
                }
            }
        });
        modelsReady.detection = true;

        setProgressText('Descargando CSRNet...');
        setProgress(65);

        try {
            crowdSession = await ort.InferenceSession.create(CROWD_MODEL_URL, {
                executionProviders: ['wasm']
            });
            modelsReady.crowd = true;
        } catch (e) {
            console.warn('CSRNet no disponible:', e.message);
        }

        setProgress(100);
        setProgressText('¡Modelos listos!');

        setTimeout(() => {
            showModelProgress(false);
            enableUpload();
        }, 500);

    } catch (err) {
        console.error('Error cargando modelos:', err);
        setProgressText('Error: ' + err.message);
    }
}

function enableUpload() {
    const label = document.getElementById('upload-label');
    const text = document.getElementById('upload-text');
    label.classList.remove('disabled');
    text.innerHTML = '<strong>Seleccionar imagen</strong><br>o arrastrar aquí';
}

// === IMAGE ANALYSIS ===

async function handleImageUpload(e) {
    const file = e.target.files[0];
    if (!file || !modelsReady.detection) return;

    hideResults();
    showAnalysisSteps(true);

    const img = new Image();
    img.onload = () => analyzeImage(img);
    img.src = URL.createObjectURL(file);
}

async function analyzeImage(img) {
    let detectionCount = null;
    let crowdCount = null;

    // Step 1: Load image
    setStep('load', 'active');
    await delay(300);
    setStep('load', 'done');

    // Step 2: DETR detection
    setStep('detect', 'active');
    try {
        const results = await detector(img.src, { threshold: 0.7 });
        const people = results.filter(r => r.label === 'person');
        detectionCount = people.length;
        drawDetections(img, results);
    } catch (err) {
        console.error('Error DETR:', err);
    }
    setStep('detect', 'done');

    // Step 3: CSRNet density
    if (crowdSession) {
        setStep('density', 'active');
        try {
            crowdCount = await runCrowdModel(img);
        } catch (err) {
            console.error('Error CSRNet:', err);
        }
        setStep('density', 'done');
    } else {
        setStep('density', 'done'); // Skip
    }

    await delay(300);
    showAnalysisSteps(false);
    showResults(detectionCount, crowdCount);
}

async function runCrowdModel(img) {
    const canvas = document.createElement('canvas');
    canvas.width = 1024;
    canvas.height = 768;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, 1024, 768);

    const imageData = ctx.getImageData(0, 0, 1024, 768);
    const { data } = imageData;

    const floatData = new Float32Array(3 * 1024 * 768);
    for (let i = 0; i < 1024 * 768; i++) {
        floatData[i] = data[i * 4] / 255;
        floatData[1024 * 768 + i] = data[i * 4 + 1] / 255;
        floatData[2 * 1024 * 768 + i] = data[i * 4 + 2] / 255;
    }

    const tensor = new ort.Tensor('float32', floatData, [1, 3, 768, 1024]);
    const output = await crowdSession.run({ input: tensor });
    const densityData = output[Object.keys(output)[0]].data;

    let count = 0;
    for (let i = 0; i < densityData.length; i++) {
        count += densityData[i];
    }
    return Math.round(count);
}

// === RESULTS ===

function showResults(detection, crowd) {
    document.getElementById('results').style.display = 'block';
    document.getElementById('count-detection').textContent = detection ?? '-';

    const rowCrowd = document.getElementById('row-crowd');
    if (crowd !== null) {
        document.getElementById('count-crowd').textContent = crowd;
        rowCrowd.style.display = 'flex';
    } else {
        rowCrowd.style.display = 'none';
    }

    let final, note;

    if (detection !== null && crowd !== null) {
        if (crowd > detection * 2) {
            final = crowd;
            note = 'Escena densa: usando estimación por densidad';
        } else if (detection > crowd) {
            final = detection;
            note = 'Personas visibles: usando detección directa';
        } else {
            final = Math.round((detection + crowd) / 2);
            note = 'Promedio de ambos métodos';
        }
    } else {
        final = detection ?? crowd ?? '-';
        note = detection !== null ? 'Solo detección disponible' : 'Solo estimación por densidad';
    }

    document.getElementById('count-total').textContent = final;
    document.getElementById('result-note').textContent = note;
}

function hideResults() {
    document.getElementById('results').style.display = 'none';
    document.getElementById('canvas-container').style.display = 'none';
}

function drawDetections(img, results) {
    const container = document.getElementById('canvas-container');
    container.innerHTML = '';
    container.style.display = 'block';

    const canvas = document.createElement('canvas');
    canvas.width = img.width;
    canvas.height = img.height;
    container.appendChild(canvas);

    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);

    results.forEach(({ label, score, box }) => {
        if (label !== 'person') return;
        const { xmin, ymin, xmax, ymax } = box;

        ctx.strokeStyle = '#1a1a2e';
        ctx.lineWidth = 2;
        ctx.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);

        ctx.font = '12px Inter, sans-serif';
        ctx.fillStyle = '#1a1a2e';
        ctx.fillText(`${Math.round(score * 100)}%`, xmin + 4, ymin + 14);
    });
}

// === UI HELPERS ===

function showModelProgress(show) {
    document.getElementById('model-progress').style.display = show ? 'block' : 'none';
}

function setProgress(pct) {
    document.getElementById('progress-fill').style.width = pct + '%';
    document.getElementById('progress-percent').textContent = pct + '%';
}

function setProgressText(text) {
    document.getElementById('progress-text').textContent = text;
}

function showAnalysisSteps(show) {
    const el = document.getElementById('analysis-steps');
    el.style.display = show ? 'block' : 'none';

    if (show) {
        // Reset all steps
        ['load', 'detect', 'density'].forEach(id => {
            const step = document.getElementById(`step-${id}`);
            const icon = document.getElementById(`step-${id}-icon`);
            step.className = 'step';
            icon.innerHTML = '○';
        });
    }
}

function setStep(id, state) {
    const step = document.getElementById(`step-${id}`);
    const icon = document.getElementById(`step-${id}-icon`);

    step.className = 'step ' + state;

    if (state === 'active') {
        icon.innerHTML = '<div class="spinner"></div>';
    } else if (state === 'done') {
        icon.innerHTML = '✓';
    }
}

function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}