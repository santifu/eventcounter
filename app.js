let objectDetector;
let img;
let canvas;
let faceApiReady = false;

// Filter state
let filters = {
    people: true,
    demographics: true,
    animals: true
};

function setup() {
    console.log("Iniciando carga de modelos...");
    updateStatus("Cargando modelos de IA...", true);
    objectDetector = ml5.objectDetector('cocossd', {}, modelLoaded);

    loadFaceApiModels();

    // File uploader
    const uploader = document.getElementById('uploader');
    uploader.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            updateStatus('Cargando imagen...', true);
            resetStats();

            const reader = new FileReader();
            reader.onload = (event) => {
                img = createImg(event.target.result, imageReady);
                img.hide();
            };
            reader.readAsDataURL(file);
        }
    });

    // Filter buttons
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
    // People box
    const boxTotal = document.getElementById('box-total');
    if (boxTotal) boxTotal.style.display = filters.people ? 'block' : 'none';

    // Demographics boxes
    const boxMen = document.getElementById('box-men');
    const boxWomen = document.getElementById('box-women');
    const boxChildren = document.getElementById('box-children');
    const showDemo = filters.demographics;
    if (boxMen) boxMen.style.display = showDemo ? 'block' : 'none';
    if (boxWomen) boxWomen.style.display = showDemo ? 'block' : 'none';
    if (boxChildren) boxChildren.style.display = showDemo ? 'block' : 'none';

    // Animals box
    const boxAnimals = document.getElementById('box-animals');
    if (boxAnimals) boxAnimals.style.display = filters.animals ? 'block' : 'none';
}

async function loadFaceApiModels() {
    try {
        await faceapi.nets.tinyFaceDetector.loadFromUri('./models');
        await faceapi.nets.faceLandmark68Net.loadFromUri('./models');
        await faceapi.nets.ageGenderNet.loadFromUri('./models');
        console.log("Modelos FaceAPI cargados.");
        faceApiReady = true;
        updateStatus("", false);
    } catch (err) {
        console.error("Error cargando modelos FaceAPI:", err);
        updateStatus("Error cargando IA: " + err.message, true);
    }
}

function modelLoaded() {
    console.log("Modelo COCO-SSD cargado.");
}

function updateStatus(msg, show) {
    const statusEl = document.getElementById('status-message');
    if (statusEl) {
        statusEl.innerText = msg;
        statusEl.style.display = show ? 'block' : 'none';
    }
}

function resetStats() {
    const statsDisplay = document.getElementById('stats-display');
    if (statsDisplay) statsDisplay.style.display = 'none';

    ['count-total', 'count-men', 'count-women', 'count-children', 'count-animals'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.innerText = '-';
    });
}

function imageReady() {
    updateStatus('Analizando imagen...', true);

    if (canvas) canvas.remove();
    canvas = createCanvas(img.width, img.height);
    canvas.parent('canvas-container');
    image(img, 0, 0);

    // Run COCO-SSD if people or animals are selected
    if (filters.people || filters.animals) {
        objectDetector.detect(img, gotCocoResult);
    } else {
        document.getElementById('stats-display').style.display = 'flex';
    }

    // Run FaceAPI if demographics are selected
    if (filters.demographics) {
        runFaceApiWithRetry();
    } else {
        // If no demographics, clear the status after COCO
        if (!filters.people && !filters.animals) {
            updateStatus('', false);
        }
    }
}

function runFaceApiWithRetry(attempts = 0) {
    if (faceApiReady) {
        detectFaces().catch(err => {
            console.error("Error en FaceAPI:", err);
        });
    } else {
        if (attempts < 5) {
            updateStatus(`Cargando IA (${attempts + 1}/5)...`, true);
            setTimeout(() => runFaceApiWithRetry(attempts + 1), 1000);
        } else {
            updateStatus("IA avanzada no disponible.", true);
        }
    }
}

function gotCocoResult(err, results) {
    if (err) {
        console.error(err);
        updateStatus('Error: ' + err.message, true);
        return;
    }

    const people = results.filter(obj => obj.label === 'person');
    const animalLabels = ['cat', 'dog', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'];
    const animals = results.filter(obj => animalLabels.includes(obj.label));

    document.getElementById('stats-display').style.display = 'flex';

    if (filters.people) {
        document.getElementById('count-total').innerText = people.length;
        drawBoxes(people, [0, 200, 100]);
    }

    if (filters.animals) {
        document.getElementById('count-animals').innerText = animals.length;
        drawBoxes(animals, [255, 165, 0]);
    }

    // Clear status if not waiting for FaceAPI
    if (!filters.demographics) {
        updateStatus('', false);
    }
}

async function detectFaces() {
    if (!img || !img.elt) {
        throw new Error("Imagen no lista");
    }

    const options = new faceapi.TinyFaceDetectorOptions({ inputSize: 416, scoreThreshold: 0.5 });
    const detections = await faceapi.detectAllFaces(img.elt, options)
        .withFaceLandmarks()
        .withAgeAndGender();

    let men = 0, women = 0, children = 0;

    detections.forEach(d => {
        const { age, gender } = d;
        if (age < 18) {
            children++;
        } else {
            if (gender === 'male') men++;
            else women++;
        }
    });

    document.getElementById('count-men').innerText = men;
    document.getElementById('count-women').innerText = women;
    document.getElementById('count-children').innerText = children;

    updateStatus('', false);
}

function drawBoxes(objects, color) {
    stroke(color[0], color[1], color[2]);
    strokeWeight(3);
    noFill();
    objects.forEach(obj => {
        rect(obj.x, obj.y, obj.width, obj.height);

        noStroke();
        fill(color[0], color[1], color[2]);
        textSize(16);
        text(obj.label, obj.x, obj.y > 10 ? obj.y - 5 : 10);

        noFill();
        stroke(color[0], color[1], color[2]);
    });
}