import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0';

env.allowLocalModels = false;

// State
let detector = null;
let crowdSession = null;
let modelsReady = { detection: false, crowd: false };

// CSRNet model URL
const CROWD_MODEL_URL = 'https://huggingface.co/muasifk/CSRNet/resolve/main/model1_A.onnx';

document.addEventListener('DOMContentLoaded', init);

async function init() {
    setupUploader();
    await loadModels();
}

function setupUploader() {
    document.getElementById('uploader').addEventListener('change', handleImageUpload);
}

async function loadModels() {
    showLoading(true);
    updateStatus('Cargando modelo de detección...');

    try {
        // Load DETR
        detector = await pipeline('object-detection', 'Xenova/detr-resnet-50', {
            progress_callback: (p) => {
                if (p.status === 'progress') {
                    setProgress(Math.round((p.loaded / p.total) * 50));
                }
            }
        });
        modelsReady.detection = true;
        console.log('DETR cargado.');

        // Try loading CSRNet (optional)
        updateStatus('Cargando modelo de densidad...');
        setProgress(60);

        try {
            crowdSession = await ort.InferenceSession.create(CROWD_MODEL_URL, {
                executionProviders: ['wasm']
            });
            modelsReady.crowd = true;
            console.log('CSRNet cargado.');
        } catch (crowdErr) {
            console.warn('CSRNet no disponible:', crowdErr.message);
            // Continue without crowd model
        }

        setProgress(100);
        showLoading(false);
        updateStatus('');
        enableUpload();

    } catch (err) {
        console.error('Error cargando modelos:', err);
        updateStatus('Error: ' + err.message);
        showLoading(false);
    }
}

function enableUpload() {
    const label = document.getElementById('upload-label');
    const text = document.getElementById('upload-text');
    label.classList.remove('disabled');
    text.innerHTML = '<strong>Seleccionar imagen</strong><br>o arrastrar aquí';
}

async function handleImageUpload(e) {
    const file = e.target.files[0];
    if (!file || !modelsReady.detection) return;

    updateStatus('Analizando imagen...');
    hideResults();

    const img = new Image();
    img.onload = () => analyzeImage(img);
    img.src = URL.createObjectURL(file);
}

async function analyzeImage(img) {
    let detectionCount = null;
    let crowdCount = null;

    // Run DETR detection
    try {
        updateStatus('Detectando personas...');
        const results = await detector(img.src, { threshold: 0.7 });
        const people = results.filter(r => r.label === 'person');
        detectionCount = people.length;
        drawDetections(img, results);
    } catch (err) {
        console.error('Error DETR:', err);
    }

    // Run CSRNet if available
    if (crowdSession) {
        try {
            updateStatus('Estimando densidad...');
            crowdCount = await runCrowdModel(img);
        } catch (err) {
            console.error('Error CSRNet:', err);
        }
    }

    // Show results
    showResults(detectionCount, crowdCount);
    updateStatus('');
}

async function runCrowdModel(img) {
    const canvas = document.createElement('canvas');
    canvas.width = 1024;
    canvas.height = 768;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, 1024, 768);

    const imageData = ctx.getImageData(0, 0, 1024, 768);
    const { data } = imageData;

    // Prepare tensor (CHW, /255)
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

function showResults(detection, crowd) {
    document.getElementById('results').style.display = 'block';

    // Detection result
    document.getElementById('count-detection').textContent =
        detection !== null ? detection : '-';

    // Crowd result
    const rowCrowd = document.getElementById('row-crowd');
    if (crowd !== null) {
        document.getElementById('count-crowd').textContent = crowd;
        rowCrowd.style.display = 'flex';
    } else {
        rowCrowd.style.display = 'none';
    }

    // Calculate final estimate
    let final;
    let note = '';

    if (detection !== null && crowd !== null) {
        // Use higher value for crowds, average for small groups
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
    } else if (detection !== null) {
        final = detection;
        note = 'Solo detección disponible';
    } else if (crowd !== null) {
        final = crowd;
        note = 'Solo estimación por densidad';
    } else {
        final = '-';
        note = 'No se pudo analizar la imagen';
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

        const text = `${Math.round(score * 100)}%`;
        ctx.font = '12px Inter, sans-serif';
        ctx.fillStyle = '#1a1a2e';
        ctx.fillText(text, xmin + 4, ymin + 14);
    });
}

function updateStatus(msg) {
    document.getElementById('status-message').textContent = msg;
}

function showLoading(show) {
    document.getElementById('loading-bar').style.display = show ? 'block' : 'none';
}

function setProgress(pct) {
    document.getElementById('loading-bar-fill').style.width = pct + '%';
}