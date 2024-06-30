from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

model_path = 'bodybuilding_pose_classifier.h5'

model = load_model(model_path)

class_labels = ['Side Chest', 'Front Double Biceps', 'Back Double Biceps', 'Front Lat Spread', 'Back Lat Spread']

def predict_pose(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    return class_labels[predicted_class[0]]

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            filepath = os.path.join('uploads', file.filename)
            file.save(filepath)
            pose = predict_pose(filepath)
            return render_template('result.html', pose=pose)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
