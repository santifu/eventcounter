let objectDetector;
let img;
let canvas;
let faceApiReady = false;

function setup() {
    // Cargar modelo COCO-SSD de ml5
    console.log("Iniciando carga de modelos...");
    updateStatus("Cargando modelo básico (COCO-SSD)...", true);
    objectDetector = ml5.objectDetector('cocossd', {}, modelLoaded);

    // Cargar modelos de FaceAPI
    loadFaceApiModels();

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
}

async function loadFaceApiModels() {
    try {
        updateStatus("Cargando inteligencia avanzada...", true);
        await faceapi.nets.tinyFaceDetector.loadFromUri('./models');
        await faceapi.nets.faceLandmark68Net.loadFromUri('./models');
        await faceapi.nets.ageGenderNet.loadFromUri('./models');
        console.log("Modelos FaceAPI cargados.");
        faceApiReady = true;
        // Si ya había terminado COCO-SSD, esto limpiará el mensaje
        const statusEl = document.getElementById('status-message');
        if (statusEl.innerText.includes("Cargando inteligencia")) {
            updateStatus("", false);
        }
    } catch (err) {
        console.error("Error cargando modelos FaceAPI:", err);
        updateStatus("Error cargando IA avanzada: " + err.message, true);
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
        console.log("Estado:", msg);
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

    // Ajustar canvas
    if (canvas) canvas.remove();
    canvas = createCanvas(img.width, img.height);
    canvas.parent('canvas-container');
    image(img, 0, 0);

    // Ejecutar COCO-SSD (Personas y Animales)
    objectDetector.detect(img, gotCocoResult);

    // Ejecutar FaceAPI con reintento si es necesario
    runFaceApiWithRetry();
}

function runFaceApiWithRetry(attempts = 0) {
    if (faceApiReady) {
        detectFaces().catch(err => {
            console.error("Error en FaceAPI:", err);
            updateStatus("Error analizando caras: " + err.message, true);
        });
    } else {
        if (attempts < 5) {
            console.warn(`FaceAPI no listo. Reintento ${attempts + 1}...`);
            updateStatus(`Cargando módulos de IA (${attempts + 1}/5)...`, true);
            setTimeout(() => runFaceApiWithRetry(attempts + 1), 1000);
        } else {
            console.error("FaceAPI Timeout.");
            updateStatus("La IA avanzada tardó demasiado en cargar. Mostrando resultados básicos.", true);
        }
    }
}

function gotCocoResult(err, results) {
    if (err) {
        console.error(err);
        updateStatus('Error en detección básica: ' + err.message, true);
        return;
    }

    const people = results.filter(obj => obj.label === 'person');
    const animals = results.filter(obj => ['cat', 'dog', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'].includes(obj.label));

    // Actualizar UI básica inmediatamente
    document.getElementById('stats-display').style.display = 'flex';
    document.getElementById('count-total').innerText = people.length;
    document.getElementById('count-animals').innerText = animals.length;

    // Dibujar recuadros
    drawBoxes(people, [0, 255, 0]); // Verde para personas
    drawBoxes(animals, [255, 165, 0]); // Naranja para animales
}

async function detectFaces() {
    // Asegurarse de que p5 haya creado el elemento DOM
    if (!img || !img.elt) {
        throw new Error("Imagen no inicializada correctamente");
    }

    console.log("Iniciando detección de caras...");

    // Usar una configuración explícita para TinyFaceDetector
    // inputSize debe ser divisible por 32. 416 es standard.
    // scoreThreshold 0.5 filtra detecciones débiles.
    const options = new faceapi.TinyFaceDetectorOptions({ inputSize: 416, scoreThreshold: 0.5 });

    const detections = await faceapi.detectAllFaces(img.elt, options)
        .withFaceLandmarks()
        .withAgeAndGender();

    console.log(`Detecciones encontradas: ${detections.length}`);

    let men = 0;
    let women = 0;
    let children = 0;

    detections.forEach(d => {
        const { age, gender } = d;
        console.log(`Cara: ${gender}, ${age.toFixed(1)} años`);

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

    // Dibujar caras (opcional, pero útil para debug)
    /* 
    const displaySize = { width: img.width, height: img.height };
    faceapi.matchDimensions(canvas.elt, displaySize);
    const resizedDetections = faceapi.resizeResults(detections, displaySize);
    faceapi.draw.drawDetections(canvas.elt, resizedDetections);
    */

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

        // Restaurar para el siguiente
        noFill();
        stroke(color[0], color[1], color[2]);
    });
}