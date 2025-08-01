import numpy as np
import joblib
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load ML models and label encoder
ml_model = joblib.load("models/voting_model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# Load DL models
cnn_model = load_model("models/cnn_model.h5")
mobilenet_model = load_model("models/mobilenet_model.h5")
resnet_model = load_model("models/resnet_model.h5")

# Load image labels (used in DL)
image_labels = sorted(['eczema', 'psoriasis', 'acne', 'ringworm', 'vitiligo'])

def predict_from_symptoms(symptom_text):
    # Preprocessing: Here we simulate symptoms preprocessing
    df = pd.read_csv("dermatology_database_updated.csv")
    all_symptoms = df.columns[:-1]  # assume last col is 'disease'
    input_features = np.zeros((1, len(all_symptoms)))

    for i, sym in enumerate(all_symptoms):
        if sym.lower() in symptom_text.lower():
            input_features[0, i] = 1

    pred_encoded = ml_model.predict(input_features)
    pred_disease = label_encoder.inverse_transform(pred_encoded)
    return pred_disease[0]

def predict_from_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.

    preds = []
    for model in [cnn_model, mobilenet_model, resnet_model]:
        p = model.predict(img_array)
        preds.append(np.argmax(p))

    # Voting among 3 DL models
    final_pred_idx = max(set(preds), key=preds.count)
    return image_labels[final_pred_idx]

def unified_predict(input_text=None, image_path=None):
    if input_text:
        return predict_from_symptoms(input_text)
    elif image_path:
        return predict_from_image(image_path)
    else:
        return "Please provide symptom text or image."