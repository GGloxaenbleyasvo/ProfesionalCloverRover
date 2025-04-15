from ultralytics import YOLO

# Load a trained model
model = YOLO("C:/Модуль Б/vladistalin-01.pt")  # Путь к вашей обученной модели

# Perform inference on an image
results = model.predict("C:/Модуль Б/datasets/train/images") # Укажите путь к изображению

# Print results (классы и вероятности)
print(results[0].probs)
