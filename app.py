from flask import Flask, request, jsonify, render_template
from gradio_client import Client, handle_file
import base64
import os
import uuid
import shutil
import json
from ultralytics import YOLO

app = Flask(__name__)

# Инициализация клиента Gradio для API
client = Client("linoyts/sketch-to-3d", hf_token="yourtoken")

# Загрузка модели YOLOv8
yolo_model = YOLO('yolov8m.pt')

# Папки для хранения файлов
TEMP_DIR = 'temp'
RESULTS_DIR = 'static/results'
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Список запрещённых промтов
FORBIDDEN_PROMPTS = ["penis", "tits", "ass"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    print("Полученные данные:", data)  # Отладка

    if not data or 'image' not in data or 'prompt' not in data or 'style' not in data:
        return jsonify({'error': 'Отсутствуют необходимые данные: image, prompt или style'}), 400

    prompt = data['prompt']
    # Проверка на запрещённые промты
    if any(word.lower() in prompt.lower() for word in FORBIDDEN_PROMPTS):
        return jsonify({'error': 'Запрещённый промт'}), 400

    try:
        image_data = data['image'].split(',')[1]  # Удаляем префикс data URL
        prompt = data['prompt']
        style = data['style']

        # Сохраняем изображение из base64
        image_bytes = base64.b64decode(image_data)
        image_path = os.path.join(TEMP_DIR, f'sketch_{uuid.uuid4()}.png')
        with open(image_path, 'wb') as f:
            f.write(image_bytes)

        # Запускаем сессию
        session = client.predict(api_name="/start_session")
        print(f"Сессия запущена: {session}")

        # Препроцессинг изображения
        result = client.predict(
            image={"background": None, "layers": [], "composite": handle_file(image_path), "id": None},
            prompt=prompt,
            negative_prompt="",
            style_name=style,
            num_steps=8,
            guidance_scale=5,
            controlnet_conditioning_scale=0.85,
            api_name="/preprocess_image"
        )
        print(f"Препроцессинг завершён: {result}")

        # Генерация 3D
        result_3d = client.predict(
            image=(handle_file(result[0]), handle_file(result[1])),
            multiimages=[],
            seed=0,
            ss_guidance_strength=7.5,
            ss_sampling_steps=12,
            slat_guidance_strength=3,
            slat_sampling_steps=12,
            multiimage_algo="stochastic",
            api_name="/image_to_3d"
        )
        print(f"3D сгенерировано: {result_3d}")

        # Сохраняем результаты
        unique_id = str(uuid.uuid4())
        image_url = f"/static/results/{unique_id}_image.webp"
        video_url = f"/static/results/{unique_id}_model.mp4"
        # Сохраняем сгенерированное изображение
        shutil.move(result[1], os.path.join(RESULTS_DIR, f"{unique_id}_image.webp"))
        shutil.move(result_3d['video'], os.path.join(RESULTS_DIR, f"{unique_id}_model.mp4"))

        # Сохраняем эскиз
        sketch_save_path = os.path.join(RESULTS_DIR, f"{unique_id}_sketch.png")
        shutil.copy(image_path, sketch_save_path)

        # Сохраняем данные
        data_to_save = {"prompt": prompt, "style": style}
        data_path = os.path.join(RESULTS_DIR, f"{unique_id}_data.json")
        with open(data_path, 'w') as f:
            json.dump(data_to_save, f)

        # Удаляем временный файл
        os.remove(image_path)

        return jsonify({'redirect': f'/result/{unique_id}'})
    except Exception as e:
        print(f"Ошибка генерации: {str(e)}")
        return jsonify({'error': f'Ошибка генерации: {str(e)}'}), 500

@app.route('/result/<unique_id>')
def result(unique_id):
    image_url = f"/static/results/{unique_id}_image.webp"
    video_url = f"/static/results/{unique_id}_model.mp4"
    return render_template('result.html', image_url=image_url, video_url=video_url, unique_id=unique_id)

@app.route('/make_toy/<unique_id>', methods=['POST'])
def make_toy(unique_id):
    # Загружаем сгенерированное изображение
    image_path = os.path.join(RESULTS_DIR, f"{unique_id}_image.webp")
    if not os.path.exists(image_path):
        return jsonify({'error': 'Изображение не найдено'}), 404

    # Проверяем наличие человека с помощью YOLOv8
    results = yolo_model(image_path)
    persons_detected = any(int(cls) == 0 for cls in results[0].boxes.cls)  # Класс 0 - человек

    if not persons_detected:
        return jsonify({'error': 'Человек на изображении не обнаружен'}), 400

    # Загружаем данные
    data_path = os.path.join(RESULTS_DIR, f"{unique_id}_data.json")
    if not os.path.exists(data_path):
        return jsonify({'error': 'Файл данных не найден'}), 404
    with open(data_path, 'r') as f:
        data = json.load(f)
    original_prompt = data['prompt']
    style = data['style']

    # Модифицируем промт
    toy_prompt = f"{original_prompt} as a toy figure"

    # Загружаем эскиз
    sketch_path = os.path.join(RESULTS_DIR, f"{unique_id}_sketch.png")
    if not os.path.exists(sketch_path):
        return jsonify({'error': 'Эскиз не найден'}), 404

    try:
        # Запускаем новую сессию
        session = client.predict(api_name="/start_session")
        print(f"Новая сессия для игрушки: {session}")

        # Генерируем изображение игрушки
        result = client.predict(
            image={"background": None, "layers": [], "composite": handle_file(sketch_path), "id": None},
            prompt=toy_prompt,
            negative_prompt="",
            style_name=style,
            num_steps=8,
            guidance_scale=5,
            controlnet_conditioning_scale=0.85,
            api_name="/preprocess_image"
        )
        print(f"Препроцессинг игрушки: {result}")

        # Сохраняем изображение игрушки
        toy_image_path = os.path.join(RESULTS_DIR, f"{unique_id}_toy.webp")
        shutil.move(result[1], toy_image_path)

        return jsonify({'toy_url': f"/static/results/{unique_id}_toy.webp"})
    except Exception as e:
        print(f"Ошибка генерации изображения игрушки: {str(e)}")
        return jsonify({'error': f'Ошибка генерации изображения игрушки: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)