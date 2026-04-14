import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("model/model.h5")


classes = ['glass', 'metal', 'paper', 'plastic','trash']

# Load image
img = Image.open("dataset/glass/4 (1).jpg")
img = img.resize((224, 224))

img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
predicted_class = classes[np.argmax(prediction)]

print("Material:", predicted_class)

if predicted_class == "trash":
    print("Status: Not Recyclable ❌")
else:
    print("Status: Recyclable ♻️")

print("Prediction:", classes[np.argmax(prediction)])