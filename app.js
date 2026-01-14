import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0';

// Configure transformers.js
env.allowLocalModels = false;

// State
let detector = null;
let filters = { people: true, animals: true };

// Animal labels from COCO dataset
const ANIMAL_LABELS = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'];

// Initialize on page load
document.addEventListener('DOMContentLoaded', init);

async function init() {
    setupFilterButtons();
    setupUploader();
    await loadModel();
}

function setupFilterButtons() {
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const filterName = btn.dataset.filter;
            filters[filterName] = !filters[filterName];
            btn.classList.toggle('active', filters[filterName]);
            updateStatBoxVisibility();
        });
    });
    updateStatBoxVisibility();
}

function updateStatBoxVisibility() {
    const boxTotal = document.getElementById('box-total');
    const boxAnimals = document.getElementById('box-animals');
    if (boxTotal) boxTotal.style.display = filters.people ? 'block' : 'none';
    if (boxAnimals) boxAnimals.style.display = filters.animals ? 'block' : 'none';
}

function setupUploader() {
    const uploader = document.getElementById('uploader');
    uploader.addEventListener('change', handleImageUpload);
}

async function loadModel() {
    updateStatus('Descargando modelo DETR (~40MB)...', true);
    showProgress(true);

    try {
        detector = await pipeline('object-detection', 'Xenova/detr-resnet-50', {
            progress_callback: (progress) => {
                if (progress.status === 'progress') {
                    const pct = Math.round((progress.loaded / progress.total) * 100);
                    setProgress(pct);
                    updateStatus(`Descargando modelo... ${pct}%`, true);
                }
            }
        });

        showProgress(false);
        updateStatus('', false);
        enableUpload();
        console.log('Modelo DETR cargado correctamente.');
    } catch (err) {
        console.error('Error cargando modelo:', err);
        updateStatus('Error cargando modelo: ' + err.message, true);
        showProgress(false);
    }
}

function enableUpload() {
    const label = document.getElementById('upload-label');
    label.classList.remove('disabled');
    label.textContent = '📷 Seleccionar Imagen';
}

async function handleImageUpload(e) {
    const file = e.target.files[0];
    if (!file || !detector) return;

    updateStatus('Analizando imagen...', true);
    resetStats();

    // Create image element
    const img = new Image();
    img.onload = async () => {
        await analyzeImage(img);
    };
    img.src = URL.createObjectURL(file);
}

async function analyzeImage(img) {
    try {
        // Run detection
        const results = await detector(img.src, { threshold: 0.7 });

        console.log('Detecciones:', results);

        // Filter and count
        const people = results.filter(r => r.label === 'person');
        const animals = results.filter(r => ANIMAL_LABELS.includes(r.label));

        // Update stats
        document.getElementById('stats-display').style.display = 'flex';

        if (filters.people) {
            document.getElementById('count-total').innerText = people.length;
        }
        if (filters.animals) {
            document.getElementById('count-animals').innerText = animals.length;
        }

        // Draw on canvas
        drawResults(img, results);

        updateStatus('', false);
    } catch (err) {
        console.error('Error en análisis:', err);
        updateStatus('Error: ' + err.message, true);
    }
}

function drawResults(img, results) {
    const container = document.getElementById('canvas-container');
    container.innerHTML = '';

    const canvas = document.createElement('canvas');
    canvas.id = 'output-canvas';
    canvas.width = img.width;
    canvas.height = img.height;
    container.appendChild(canvas);

    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);

    results.forEach(result => {
        const { label, score, box } = result;
        const { xmin, ymin, xmax, ymax } = box;

        // Determine if we should draw this
        const isPerson = label === 'person';
        const isAnimal = ANIMAL_LABELS.includes(label);

        if ((isPerson && !filters.people) || (isAnimal && !filters.animals)) {
            return;
        }
        if (!isPerson && !isAnimal) {
            return; // Skip other objects
        }

        // Colors
        const color = isPerson ? '#00c853' : '#ff9800';

        // Draw box
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);

        // Draw label
        const labelText = `${label} ${Math.round(score * 100)}%`;
        ctx.font = 'bold 14px Arial';
        const textWidth = ctx.measureText(labelText).width;

        ctx.fillStyle = color;
        ctx.fillRect(xmin, ymin - 22, textWidth + 10, 22);

        ctx.fillStyle = 'white';
        ctx.fillText(labelText, xmin + 5, ymin - 6);
    });
}

function resetStats() {
    document.getElementById('stats-display').style.display = 'none';
    document.getElementById('count-total').innerText = '-';
    document.getElementById('count-animals').innerText = '-';

    const container = document.getElementById('canvas-container');
    container.innerHTML = '';
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