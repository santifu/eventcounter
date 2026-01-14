let objectDetector;
let img;
let canvas;

function setup() {
    // Creamos el detector usando COCO-SSD (Personas, objetos, etc.)
    objectDetector = ml5.objectDetector('cocossd', {}, modelLoaded);
    const uploader = document.getElementById('uploader');

    uploader.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            // Mostrar estado de carga
            document.getElementById('counter').innerText = 'Cargando imagen...';

            const reader = new FileReader();
            reader.onload = (event) => {
                img = createImg(event.target.result, imageReady);
                img.hide();
            };
            reader.readAsDataURL(file);
        }
    });
}

function modelLoaded() {
    console.log("Modelo cargado correctamente.");
}

function imageReady() {
    // Mostrar estado de análisis
    document.getElementById('counter').innerText = 'Analizando imagen...';

    // Ajustar el canvas al tamaño de la imagen
    if (canvas) canvas.remove();
    canvas = createCanvas(img.width, img.height);
    canvas.parent('canvas-container');
    image(img, 0, 0);

    // Ejecutar detección
    objectDetector.detect(img, gotResult);
}

function gotResult(err, results) {
    if (err) {
        console.error(err);
        return;
    }

    // Filtrar solo los objetos que son personas
    const persons = results.filter(obj => obj.label === 'person');

    document.getElementById('counter').innerText = `Personas detectadas: ${persons.length}`;

    // Dibujar recuadros
    persons.forEach(person => {
        stroke(0, 255, 0);
        strokeWeight(3);
        noFill();
        rect(person.x, person.y, person.width, person.height);

        noStroke();
        fill(0, 255, 0);
        textSize(16);
        text("Asistente", person.x, person.y > 10 ? person.y - 5 : 10);
    });
}