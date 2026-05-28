from fastapi import FastAPI, HTTPException, File, UploadFile 
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image 
import numpy as np 
import tensorflow as tf 
import io

# -----------------------------
# Inicializar app
# -----------------------------
app = FastAPI(title="API Predicción MNIST")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Cargar modelo
# -----------------------------
try:
    modelo = tf.keras.models.load_model("model/modelo_menist.keras")
except Exception as e:
    raise RuntimeError(f"Error cargando el modelo: {e}")


def get_bounding_box(img):
    coords = np.column_stack(np.where(img > 0))
    if coords.size == 0:
        return None
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return y_min, y_max, x_min, x_max


def center_of_mass_shift(img):
    total = np.sum(img)
    if total == 0:
        return img

    indices = np.indices(img.shape)
    cy = np.sum(indices[0] * img) / total
    cx = np.sum(indices[1] * img) / total

    shift_y = int(np.round(img.shape[0] / 2 - cy))
    shift_x = int(np.round(img.shape[1] / 2 - cx))

    img = np.roll(img, shift_y, axis=0)
    img = np.roll(img, shift_x, axis=1)

    return img


def preprocess(image: Image.Image):
    # 1. Escala de grises
    img = image.convert("L")
    img = np.array(img)

    # 2. Invertir
    img = 255 - img

    # 3. Binarizar
    img = (img > 50).astype(np.uint8) * 255

    # 4. Bounding box
    bbox = get_bounding_box(img)
    if bbox is None:
        raise ValueError("Imagen vacía")

    y_min, y_max, x_min, x_max = bbox
    img = img[y_min:y_max, x_min:x_max]

    # 5. Redimensionar a 20x20
    img_pil = Image.fromarray(img)
    img_pil = img_pil.resize((20, 20), Image.LANCZOS)

    img = np.array(img_pil)

    # 6. Crear lienzo 28x28
    new_img = np.zeros((28, 28), dtype=np.uint8)

    x_offset = 4
    y_offset = 4

    new_img[y_offset:y_offset+20, x_offset:x_offset+20] = img

    # 7. Centrar por masa
    new_img = center_of_mass_shift(new_img)

    # 8. Normalizar
    new_img = new_img / 255.0
    new_img = new_img.astype(np.float32)

    # DEBUG
    Image.fromarray((new_img * 255).astype(np.uint8)).save("debug.png")

    return new_img


# -----------------------------
# Ruta raíz
# -----------------------------
@app.get("/")
def root():
    return {
        "message": "Modelo listo",
        "version": "2.0 PRO"
    }


# -----------------------------
# Predicción
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        image = preprocess(image)

        # Ajustar shape
        image = np.expand_dims(image, axis=-1)   # (28,28,1)
        image = np.expand_dims(image, axis=0)    # (1,28,28,1)

        print("Shape:", image.shape)
        print("Min:", np.min(image), "Max:", np.max(image))
        print("Sum:", np.sum(image))

        pred = modelo.predict(image)
        clase_idx = int(np.argmax(pred[0]))
        probabilidad = float(np.max(pred[0]))

        return {
            "clase": str(clase_idx),
            "probabilidad": probabilidad
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))