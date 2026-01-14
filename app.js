let objectDetector;
let img;
let canvas;
let faceApiReady = false;

function setup() {
    // Cargar modelo COCO-SSD de ml5
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
        await faceapi.nets.tinyFaceDetector.loadFromUri('./models');
        await faceapi.nets.faceLandmark68Net.loadFromUri('./models');
        await faceapi.nets.ageGenderNet.loadFromUri('./models');
        console.log("Modelos FaceAPI cargados.");
        faceApiReady = true;
    } catch (err) {
        console.error("Error cargando modelos FaceAPI:", err);
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

    // Ajustar canvas
    if (canvas) canvas.remove();
    canvas = createCanvas(img.width, img.height);
    canvas.parent('canvas-container');
    image(img, 0, 0);

    // Ejecutar COCO-SSD (Personas y Animales)
    objectDetector.detect(img, gotCocoResult);

    // Ejecutar FaceAPI (Género y Edad) con manejo de errores
    if (faceApiReady) {
        detectFaces().catch(err => {
            console.error("Error en FaceAPI:", err);
            // No bloqueamos, solo mostramos en consola
        });
    } else {
        console.warn("FaceAPI aún no está listo. Reintentando en 1s...");
        setTimeout(() => {
            if (faceApiReady) detectFaces();
        }, 1000);
    }
}

function gotCocoResult(err, results) {
    if (err) {
        console.error(err);
        updateStatus('Error en detección: ' + err.message, true);
        return;
    }

    const people = results.filter(obj => obj.label === 'person');
    const animals = results.filter(obj => ['cat', 'dog', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'].includes(obj.label));

    // Actualizar UI
    document.getElementById('stats-display').style.display = 'flex';
    document.getElementById('count-total').innerText = people.length;
    document.getElementById('count-animals').innerText = animals.length;

    // Dibujar recuadros
    drawBoxes(people, [0, 255, 0]); // Verde para personas
    drawBoxes(animals, [255, 165, 0]); // Naranja para animales

    // Si ya no estamos esperando a FaceAPI (o falló), quitamos el loading
    // Pero lo ideal es esperar a ambos. Haremos que detectFaces limpie el status.
}

async function detectFaces() {
    // Usar el elemento DOM de la imagen creado por p5 (img.elt)
    // TinyFaceDetectorOptions: inputSize y scoreThreshold se pueden ajustar
    const detections = await faceapi.detectAllFaces(img.elt, new faceapi.TinyFaceDetectorOptions({ inputSize: 416 }))
        .withFaceLandmarks()
        .withAgeAndGender();

    let men = 0;
    let women = 0;
    let children = 0;

    detections.forEach(d => {
        const { age, gender } = d;
        // console.log(`Detected: ${gender}, ${age}`);

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
    // faceapi.draw.drawDetections(canvas.elt, detections);

    updateStatus('', false); // Ocultar mensaje de estado al terminar todo
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