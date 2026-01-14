import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0';

env.allowLocalModels = false;

let detector = null;
let crowdSession = null;

const CROWD_MODEL_URL = 'https://huggingface.co/muasifk/CSRNet/resolve/main/model1_A.onnx';

// Object categories
const CATEGORIES = {
    person: { label: 'Personas', icon: '👤' },
    dog: { label: 'Perros', icon: '🐕' },
    cat: { label: 'Gatos', icon: '🐱' },
    bird: { label: 'Pájaros', icon: '🐦' },
    horse: { label: 'Caballos', icon: '🐴' },
    cow: { label: 'Vacas', icon: '🐄' },
    sheep: { label: 'Ovejas', icon: '🐑' },
    car: { label: 'Coches', icon: '🚗' },
    bicycle: { label: 'Bicis', icon: '🚲' },
    motorcycle: { label: 'Motos', icon: '🏍️' }
};

document.addEventListener('DOMContentLoaded', init);

async function init() {
    setupUploader();
    await loadModels();
}

function setupUploader() {
    document.getElementById('uploader').addEventListener('change', handleImageUpload);
}

async function loadModels() {
    setProgress(0, 'Descargando modelo DETR...');

    try {
        detector = await pipeline('object-detection', 'Xenova/detr-resnet-50', {
            progress_callback: (p) => {
                if (p.status === 'progress' && p.total) {
                    const pct = Math.round((p.loaded / p.total) * 70);
                    setProgress(pct, 'Descargando DETR... ' + pct + '%');
                }
            }
        });

        setProgress(75, 'Descargando modelo de densidad...');

        try {
            crowdSession = await ort.InferenceSession.create(CROWD_MODEL_URL, {
                executionProviders: ['wasm']
            });
            setProgress(95, 'CSRNet cargado');
        } catch (e) {
            console.warn('CSRNet no disponible:', e.message);
            setProgress(95, 'Solo detección disponible');
        }

        setProgress(100, '¡Listo!');

        setTimeout(() => {
            hideProgress();
            enableUpload();
        }, 800);

    } catch (err) {
        console.error('Error:', err);
        setProgress(0, 'Error: ' + err.message);
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
    if (!file || !detector) return;

    hideResults();
    showStatus('<span class="spinner"></span> Analizando imagen...');

    const img = new Image();
    img.onload = () => analyzeImage(img);
    img.src = URL.createObjectURL(file);
}

async function analyzeImage(img) {
    let detectionResults = [];
    let crowdCount = null;

    // DETR detection
    try {
        showStatus('<span class="spinner"></span> Detectando objetos...');
        detectionResults = await detector(img.src, { threshold: 0.6 });
    } catch (err) {
        console.error('DETR error:', err);
    }

    // CSRNet density
    if (crowdSession) {
        try {
            showStatus('<span class="spinner"></span> Estimando densidad...');
            crowdCount = await runCrowdModel(img);
        } catch (err) {
            console.error('CSRNet error:', err);
        }
    }

    hideStatus();
    showResults(detectionResults, crowdCount);
    drawDetections(img, detectionResults);
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
    for (let i = 0; i < densityData.length; i++) count += densityData[i];
    return Math.round(count);
}

function showResults(results, crowdCount) {
    // Count by category
    const counts = {};
    results.forEach(r => {
        counts[r.label] = (counts[r.label] || 0) + 1;
    });

    // Build grid HTML
    const grid = document.getElementById('result-grid');
    grid.innerHTML = '';

    Object.keys(counts).forEach(label => {
        const cat = CATEGORIES[label] || { label: label, icon: '📦' };
        const item = document.createElement('div');
        item.className = 'result-item';
        item.innerHTML = `
            <div class="result-item-value">${counts[label]}</div>
            <div class="result-item-label">${cat.icon} ${cat.label}</div>
        `;
        grid.appendChild(item);
    });

    // Add crowd estimate if available
    if (crowdCount !== null) {
        const item = document.createElement('div');
        item.className = 'result-item';
        item.style.background = '#fef3c7';
        item.innerHTML = `
            <div class="result-item-value">${crowdCount}</div>
            <div class="result-item-label">👥 Densidad</div>
        `;
        grid.appendChild(item);
    }

    // Total people
    const personCount = counts['person'] || 0;
    let finalCount = personCount;
    let note = '';

    if (crowdCount !== null && crowdCount > personCount * 2) {
        finalCount = crowdCount;
        note = 'Usando estimación por densidad (escena densa)';
    } else if (personCount > 0) {
        note = 'Usando detección directa';
    }

    document.getElementById('count-total').textContent = finalCount;
    document.getElementById('result-note').textContent = note;
    document.getElementById('results').style.display = 'block';
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

    const colors = {
        person: '#22c55e',
        dog: '#f59e0b',
        cat: '#8b5cf6',
        default: '#64748b'
    };

    results.forEach(({ label, score, box }) => {
        const { xmin, ymin, xmax, ymax } = box;
        const color = colors[label] || colors.default;

        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);

        const text = `${label} ${Math.round(score * 100)}%`;
        ctx.font = 'bold 11px Inter, sans-serif';
        const textWidth = ctx.measureText(text).width;

        ctx.fillStyle = color;
        ctx.fillRect(xmin, ymin - 18, textWidth + 8, 18);

        ctx.fillStyle = '#fff';
        ctx.fillText(text, xmin + 4, ymin - 5);
    });
}

function hideResults() {
    document.getElementById('results').style.display = 'none';
    document.getElementById('canvas-container').style.display = 'none';
}

function setProgress(pct, text) {
    document.getElementById('progress-fill').style.width = pct + '%';
    document.getElementById('progress-percent').textContent = pct + '%';
    document.getElementById('progress-text').textContent = text;
}

function hideProgress() {
    document.getElementById('progress-section').classList.add('hidden');
}

function showStatus(html) {
    const el = document.getElementById('status-text');
    el.innerHTML = html;
    el.classList.remove('hidden');
}

function hideStatus() {
    document.getElementById('status-text').classList.add('hidden');
}