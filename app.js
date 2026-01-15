import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0';

env.allowLocalModels = false;

let detector = null;
let crowdSession = null;
let clipSession = null;
let clipTokenizer = null;
let faceApiReady = false;
let currentImage = null; // Store current image for recalculation

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
    document.getElementById('recalc-btn').addEventListener('click', () => {
        if (currentImage) analyzeImage(currentImage);
    });
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

        // 4. CLIP (Deep Analysis)
        // We load this lazily or here? Let's load tokenizer here, model is big.
        // Actually, let's load it here to be ready.
        setProgress(90, 'Loading CLIP (Deep Analysis)...');
        try {
            const { AutoTokenizer, AutoProcessor, CLIPTextModelWithProjection, CLIPVisionModelWithProjection } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0');

            // We'll use a wrapper or just the pipeline if available.
            // Pipeline 'zero-shot-image-classification' is easiest.
            clipSession = await pipeline('zero-shot-image-classification', 'Xenova/clip-vit-base-patch32', {
                progress_callback: (p) => {
                    if (p.status === 'progress' && p.total) {
                        // Optional progress update
                    }
                }
            });
        } catch (e) {
            console.warn('CLIP failed:', e);
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
    img.onload = () => {
        currentImage = img;
        document.getElementById('recalc-container').style.display = 'block';
        analyzeImage(img);
    };
    img.src = URL.createObjectURL(file);
}

async function analyzeImage(img) {
    // Check toggles
    const useCrowd = document.getElementById('toggle-crowd').checked;
    const useFace = document.getElementById('toggle-face').checked;
    const useClip = document.getElementById('toggle-clip').checked;

    let detectorResults = [];
    let crowdCount = null;
    let faceResults = [];
    let clipResults = null;

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

    // 4. CLIP Analysis
    if (useClip && clipSession && detectorResults.length > 0) {
        showStatus('<span class="spinner"></span> Deep Analysis (CLIP)... this may take a while');
        try {
            clipResults = await runClipAnalysis(img, detectorResults);
        } catch (err) {
            console.error('CLIP error:', err);
        }
    }

    hideStatus();
    showResults(detectorResults, crowdCount, faceResults, clipResults, useCrowd);
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

async function runClipAnalysis(img, detections) {
    // Filter for persons
    const persons = detections.filter(d => d.label === 'person');
    if (persons.length === 0) return null;

    let stats = { men: 0, women: 0, child: 0, total: 0 };
    const classes = ['man', 'woman', 'child'];

    // Create a canvas to crop images
    const canvas = document.createElement('canvas');
    canvas.width = img.width;
    canvas.height = img.height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);

    // Limit to first 20 people to avoid browser crash/hang on huge crowds
    const limit = Math.min(persons.length, 20);

    for (let i = 0; i < limit; i++) {
        const box = persons[i].box;
        const w = box.xmax - box.xmin;
        const h = box.ymax - box.ymin;

        // Skip tiny boxes
        if (w < 20 || h < 20) continue;

        try {
            // Get crop
            const cropData = ctx.getImageData(box.xmin, box.ymin, w, h);
            // Create a temporary canvas for the crop to pass as image URL or blob
            const cropCanvas = document.createElement('canvas');
            cropCanvas.width = w;
            cropCanvas.height = h;
            cropCanvas.getContext('2d').putImageData(cropData, 0, 0);
            const cropUrl = cropCanvas.toDataURL();

            // Run CLIP
            const output = await clipSession(cropUrl, classes);
            // output usually sorted by score. output[0] is best match.
            const best = output[0].label;

            if (best === 'man') stats.men++;
            else if (best === 'woman') stats.women++;
            else if (best === 'child') stats.child++;

            stats.total++;

            // Allow UI update
            await new Promise(r => setTimeout(r, 10));
            showStatus(`<span class="spinner"></span> Deep Analysis: ${i + 1}/${limit} persons...`);

        } catch (e) {
            console.warn('CLIP crop error', e);
        }
    }

    return stats;
}

function showResults(detrResults, crowdCount, faceResults, clipResults, useCrowd) {
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

    // -- Gender Breakdown in Grid --
    if (faceResults.length > 0) {
        let males = 0;
        let females = 0;
        faceResults.forEach(f => f.gender === 'male' ? males++ : females++);

        if (males > 0) {
            const item = document.createElement('div');
            item.className = 'result-item';
            // Light blue background for men
            item.style.background = '#eff6ff';
            item.innerHTML = `
                <div class="result-item-value">${males}</div>
                <div class="result-item-label">👨 Men</div>
            `;
            grid.appendChild(item);
        }
        if (females > 0) {
            const item = document.createElement('div');
            item.className = 'result-item';
            // Light pink background for women
            item.style.background = '#fdf2f8';
            item.innerHTML = `
                <div class="result-item-value">${females}</div>
                <div class="result-item-label">👩 Women</div>
            `;
            grid.appendChild(item);
        }
    }

    // -- CLIP Breakdown --
    if (clipResults && clipResults.total > 0) {
        // Add divider or distinct style
        const item = document.createElement('div');
        item.className = 'result-item';
        item.style.background = '#e0e7ff'; // Indigo
        item.style.gridColumn = "1 / -1"; // Full width
        item.innerHTML = `
            <div style="font-size:0.8rem; font-weight:bold; margin-bottom:4px;">Deep Analysis (Body)</div>
            <div style="display:flex; justify-content:space-around; font-size:0.9rem;">
                <span>👨 ${clipResults.men}</span>
                <span>👩 ${clipResults.women}</span>
                <span>👶 ${clipResults.child}</span>
            </div>
            <div style="font-size:0.6rem; color:#4338ca; margin-top:2px;">Sample: ${clipResults.total} persons</div>
        `;
        grid.appendChild(item);
    }

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
        // If toggle was on but no faces found
        if (document.getElementById('toggle-face').checked) {
            demoSection.style.display = 'block';
            demoList.innerHTML = `<div class="result-list-item" style="color:#d97706">
                ⚠️ Demographics enabled but no faces detected clearly.
            </div>`;
        } else {
            demoSection.style.display = 'none';
        }
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