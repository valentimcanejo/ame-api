import os
from ultralytics import YOLO
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import cv2
from google.cloud import vision
import numpy as np
import concurrent.futures
from werkzeug.utils import secure_filename

current_dir = os.getcwd()
credential_path = os.path.join(current_dir, 'ame1-428910-5432988dfc2b.json')
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credential_path

client = vision.ImageAnnotatorClient()

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


def read_text_from_image(image):
    _, image_encoded = cv2.imencode('.png', image)
    content = image_encoded.tobytes()
    image = vision.Image(content=content)
    request = vision.AnnotateImageRequest(
        image=image,
        features=[vision.Feature(type=vision.Feature.Type.TEXT_DETECTION)]
    )
    response = client.annotate_image(request=request)
    text_annotations = response.text_annotations
    if text_annotations:
        return text_annotations[0].description.strip()
    else:
        return ""


IMAGES_DIR = os.path.join('.', 'images')
image_path = os.path.join(IMAGES_DIR, 'pag.jpg')
model_path = os.path.join('.', 'runs', 'detect',
                          'train12', 'weights', 'last.pt')
threshold = 0.5


def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    return image


def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at path: {model_path}")
    return YOLO(model_path)


def detect_objects(image, model, image_name):
    print(f"Sending image {image_name} to Vertex AI for object detection...")
    results = model(image)
    return results


def draw_detections(image, results, threshold):
    for result in results:
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                score = scores[i]
                class_id = int(class_ids[i])
                label = result.names[class_id]
                if score > threshold and label == "Preenchido":
                    cv2.rectangle(image, (int(x1), int(y1)),
                                  (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(image, label, (int(x1+40), int(y1+9)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    return image


def resize_image(image, max_width, max_height):
    height, width = image.shape[:2]
    scaling_factor = min(max_width / width, max_height / height)
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)
    return cv2.resize(image, (new_width, new_height))


def obter_textos(image, coords):
    recortes = []
    for (x, y, largura, altura) in coords:
        # Ajuste os valores conforme necessário para pegar a área ao lado dos campos preenchidos
        margem = 10  # Margem para pegar um pouco além do campo preenchido
        nova_x = x + largura + margem
        nova_largura = 550  # Largura do recorte ao lado do campo preenchido
        nova_altura = altura  # Altura igual ao campo preenchido

        recorte = image[y:y + nova_altura, nova_x:nova_x + nova_largura]
        recortes.append(recorte)

    textos_extraidos = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        textos_extraidos = list(executor.map(read_text_from_image, recortes))

    textos_limpos = [texto.strip() for texto in textos_extraidos]
    return textos_limpos


@app.route('/upload', methods=['POST'])
def main():
    try:
        if 'file' not in request.files:
            return 'No image file found', 400

        image_file = request.files['file']
        filename = secure_filename(image_file.filename)
        image_path = os.path.join(IMAGES_DIR, filename)
        image_file.save(image_path)
        image = load_image(image_path)
        model = load_model(model_path)
        results = detect_objects(image, model, filename)

        detected_classes = set()
        pixels_preenchidos = []  # Lista para armazenar coordenadas dos campos preenchidos

        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()
                for i, (x1, y1, x2, y2) in enumerate(boxes):
                    score = scores[i]
                    class_id = int(class_ids[i])
                    label = result.names[class_id]
                    if label == "Preenchido":  # Filtrar apenas detecções de "Preenchido"
                        detected_classes.add(label)
                        print(f"Detected: {label} (score: {
                              score:.2f}) at [{x1}, {y1}, {x2}, {y2}]")

                        # Adicionar coordenadas do campo preenchido (ajuste as coordenadas conforme necessário)
                        largura = int(x2 - x1)
                        altura = int(y2 - y1)
                        pixels_preenchidos.append(
                            (int(x1), int(y1), largura, altura))

        print(f"Detected classes: {detected_classes}")

        image_with_detections = draw_detections(
            image.copy(), results, threshold)

        max_width = 1000
        max_height = 800
        resized_image = resize_image(
            image_with_detections, max_width, max_height)

        cv2.imshow('Detected Objects', resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        textos = obter_textos(image, pixels_preenchidos)
        print(textos)
        for indice, texto in enumerate(textos):
            print(f"Texto extraído {indice + 1}: {texto}")
            print("-----")
        return jsonify(textos)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    app.run(debug=True)
