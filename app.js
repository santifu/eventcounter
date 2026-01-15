import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0';

env.allowLocalModels = false;

let detector = null;
let crowdSession = null;
let faceApiReady = false;

const CROWD_MODEL_URL = 'https://huggingface.co/muasifk/CSRNet/resolve/main/model1_A.onnx';

const CATEGORIES = {
    person: { label: 'People', icon: '👤' },
    dog: { label: 'Dogs', icon: '🐕' },
    cat: { label: 'Cats', icon: '🐱' },
    bird: { label: 'Birds', icon: '🐦' },
    horse: { label: 'Horses', icon: '🐴' },
    cow: { label: 'Cows', icon: '🐄' },
    sheep: { label: 'Sheep', icon: '🐑' },
    car: { label: 'Cars', icon: '🚗' },
    bicycle: { label: 'Bicycles', icon: '🚲' },
    motorcycle: { label: 'Motorcycles', icon: '🏍️' }
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
    setProgress(0, 'Loading DETR model...');

    try {
        // 1. DETR (Base detection)
        detector = await pipeline('object-detection', 'Xenova/detr-resnet-50', {
            progress_callback: (p) => {
                if (p.status === 'progress' && p.total) {
                    const pct = Math.round((p.loaded / p.total) * 50);
                    setProgress(pct, 'Loading DETR... ' + pct + '%');
                }
            }
        });

        // 2. CSRNet (Crowd)
        setProgress(50, 'Loading Crowd Density model...');
        try {
            crowdSession = await ort.InferenceSession.create(CROWD_MODEL_URL, {
                executionProviders: ['wasm']
            });
        } catch (e) {
            console.warn('CSRNet not available:', e);
        }

        // 3. FaceAPI (Demographics)
        setProgress(80, 'Loading Face Recognition...');
        try {
            await faceapi.nets.tinyFaceDetector.loadFromUri('./models');
            await faceapi.nets.ageGenderNet.loadFromUri('./models');
            // We don't need landmarks for just age/gender if using simple detector, 
            // but ageGenderNet often requires aligned faces, so landmarks are good
            await faceapi.nets.faceLandmark68Net.loadFromUri('./models');
            faceApiReady = true;
        } catch (e) {
            console.warn('FaceAPI not available:', e);
        }

        setProgress(100, 'Ready!');
        setTimeout(() => {
            hideProgress();
            enableUpload();
        }, 800);

    } catch (err) {
        console.error('Error loading models:', err);
        setProgress(0, 'Error: ' + err.message);
    }
}

function enableUpload() {
    const label = document.getElementById('upload-label');
    const text = document.getElementById('upload-text');
    label.classList.remove('disabled');
    text.innerHTML = '<strong>Select Image</strong><br>or drop here';
}

async function handleImageUpload(e) {
    const file = e.target.files[0];
    if (!file || !detector) return;

    hideResults();
    showStatus('<span class="spinner"></span> Analyzing image...');

    const img = new Image();
    img.onload = () => analyzeImage(img);
    img.src = URL.createObjectURL(file);
}

async function analyzeImage(img) {
    // Check toggles
    const useCrowd = document.getElementById('toggle-crowd').checked;
    const useFace = document.getElementById('toggle-face').checked;

    let detectorResults = [];
    let crowdCount = null;
    let faceResults = [];

    // 1. DETR (Always run)
    try {
        showStatus('<span class="spinner"></span> Detecting objects (DETR)...');
        detectorResults = await detector(img.src, { threshold: 0.6 });
    } catch (err) {
        console.error('DETR error:', err);
    }

    // 2. CSRNet (Optional)
    if (useCrowd && crowdSession) {
        try {
            showStatus('<span class="spinner"></span> Estimating density (CSRNet)...');
            crowdCount = await runCrowdModel(img);
        } catch (err) {
            console.error('CSRNet error:', err);
        }
    }

    // 3. FaceAPI (Optional)
    if (useFace && faceApiReady) {
        try {
            showStatus('<span class="spinner"></span> Analyzing demographics (FaceAPI)...');
            // FaceAPI needs the image element in DOM often or tensor, but image object works usually
            // detectAllFaces uses SSD or TinyFace. TinyFace is faster.
            const detections = await faceapi.detectAllFaces(img, new faceapi.TinyFaceDetectorOptions())
                .withFaceLandmarks()
                .withAgeAndGender();
            faceResults = detections;
        } catch (err) {
            console.error('FaceAPI error:', err);
        }
    }

    hideStatus();
    showResults(detectorResults, crowdCount, faceResults, useCrowd);
    drawDetections(img, detectorResults, faceResults);
}

async function runCrowdModel(img) {
    const canvas = document.createElement('canvas');
    canvas.width = 1024;
    canvas.height = 768;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, 1024, 768);

    const { data } = ctx.getImageData(0, 0, 1024, 768);
    const floatData = new Float32Array(3 * 1024 * 768);

    // Normalization /255, CHW
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

function showResults(detrResults, crowdCount, faceResults, useCrowd) {
    const grid = document.getElementById('result-grid');
    grid.innerHTML = '';

    // -- DETR counts --
    const counts = {};
    detrResults.forEach(r => counts[r.label] = (counts[r.label] || 0) + 1);

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

    // -- Crowd count --
    if (crowdCount !== null) {
        const item = document.createElement('div');
        item.className = 'result-item';
        item.style.background = '#fef3c7'; // Highlight
        item.innerHTML = `
            <div class="result-item-value">${crowdCount}</div>
            <div class="result-item-label">👥 Density</div>
            <div style="font-size:0.6rem; color:#b45309">Model: CSRNet</div>
        `;
        grid.appendChild(item);
    }

    // -- Demographics --
    const demoSection = document.getElementById('demographics-section');
    const demoList = document.getElementById('demographics-list');
    demoList.innerHTML = '';

    if (faceResults.length > 0) {
        demoSection.style.display = 'block';

        let males = 0;
        let females = 0;
        let totalAge = 0;

        faceResults.forEach(f => {
            if (f.gender === 'male') males++; else females++;
            totalAge += f.age;
        });

        const avgAge = Math.round(totalAge / faceResults.length);

        demoList.innerHTML = `
            <div class="result-list-item">
                <span>Total Faces Detected</span>
                <strong>${faceResults.length}</strong>
            </div>
            <div class="result-list-item">
                <span>Average Age</span>
                <strong>${avgAge} years</strong>
            </div>
            <div class="result-list-item">
                <span>Gender Split</span>
                <strong>${males}M / ${females}F</strong>
            </div>
        `;
    } else {
        demoSection.style.display = 'none';
    }

    // -- Final Count Logic --
    const personCount = counts['person'] || 0;
    let finalCount = personCount;
    let note = '';

    if (useCrowd && crowdCount !== null) {
        if (crowdCount > personCount * 1.5) {
            finalCount = crowdCount;
            note = 'Using CSRNet (Crowd settings enabled)';
        } else {
            finalCount = personCount;
            note = 'Using DETR (Detection count similar/higher than density)';
        }
    } else {
        note = 'Using DETR (Direct detection)';
    }

    // Override note if FaceAPI found more faces? Unlikely usually, but possible.
    if (faceResults.length > finalCount) {
        finalCount = faceResults.length;
        note = 'Using FaceAPI (Detected more faces than bodies)';
    }

    document.getElementById('count-total').textContent = finalCount;
    document.getElementById('result-note').textContent = note;
    document.getElementById('results').style.display = 'block';
}

function drawDetections(img, detrResults, faceResults) {
    const container = document.getElementById('canvas-container');
    container.innerHTML = '';
    container.style.display = 'block';

    const canvas = document.createElement('canvas');
    canvas.width = img.width;
    canvas.height = img.height;
    container.appendChild(canvas);

    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);

    // Draw DETR boxes
    const colors = { person: '#22c55e', dog: '#f59e0b', default: '#64748b' };

    detrResults.forEach(({ label, score, box }) => {
        const { xmin, ymin, xmax, ymax } = box;
        const color = colors[label] || colors.default;

        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);

        // Label
        /*
        const text = `${label} ${Math.round(score * 100)}%`;
        ctx.font = 'bold 11px Inter, sans-serif';
        const w = ctx.measureText(text).width;
        ctx.fillStyle = color;
        ctx.fillRect(xmin, ymin - 18, w + 8, 18);
        ctx.fillStyle = '#fff';
        ctx.fillText(text, xmin + 4, ymin - 5);
        */
    });

    // Draw FaceAPI boxes (Blue)
    faceResults.forEach(f => {
        const { x, y, width, height } = f.detection.box;
        ctx.strokeStyle = '#3b82f6';
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, width, height);

        const text = `${f.gender} (${Math.round(f.age)})`;
        ctx.font = 'bold 11px Inter, sans-serif';
        ctx.fillStyle = '#3b82f6';
        ctx.fillRect(x, y - 18, ctx.measureText(text).width + 8, 18);
        ctx.fillStyle = '#fff';
        ctx.fillText(text, x + 4, y - 5);
    });
}

// UI HELPERS
function setProgress(pct, text) {
    document.getElementById('progress-fill').style.width = pct + '%';
    document.getElementById('progress-percent').textContent = pct + '%';
    document.getElementById('progress-text').textContent = text;
}
function hideProgress() { document.getElementById('progress-section').classList.add('hidden'); }
function showStatus(html) {
    const el = document.getElementById('status-text');
    el.innerHTML = html;
    el.classList.remove('hidden');
}
function hideStatus() { document.getElementById('status-text').classList.add('hidden'); }
function hideResults() {
    document.getElementById('results').style.display = 'none';
    document.getElementById('canvas-container').style.display = 'none';
}