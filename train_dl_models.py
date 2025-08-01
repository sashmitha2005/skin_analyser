print("Running...")
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping

# Set paths
DATASET_DIR = 'Split_smol'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# Load and preprocess data
datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)
train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)
val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

num_classes = train_gen.num_classes

# --- 1. CNN Model from scratch ---
def build_cnn_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

print("Training CNN model...")
cnn_model = build_cnn_model()
cnn_model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)
cnn_model.save('models/cnn_model.h5')

# --- 2. MobileNetV2 ---
def build_mobilenet():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    preds = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=preds)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

print("Training MobileNet model...")
mobilenet_model = build_mobilenet()
mobilenet_datagen = ImageDataGenerator(validation_split=0.2, preprocessing_function=mobilenet_preprocess)
train_mg = mobilenet_datagen.flow_from_directory(DATASET_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', subset='training')
val_mg = mobilenet_datagen.flow_from_directory(DATASET_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', subset='validation')
mobilenet_model.fit(train_mg, validation_data=val_mg, epochs=EPOCHS)
mobilenet_model.save('models/mobilenet_model.h5')

# --- 3. ResNet50 ---
def build_resnet():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    preds = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=preds)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

print("Training ResNet model...")
resnet_model = build_resnet()
resnet_datagen = ImageDataGenerator(validation_split=0.2, preprocessing_function=resnet_preprocess)
train_rg = resnet_datagen.flow_from_directory(DATASET_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', subset='training')
val_rg = resnet_datagen.flow_from_directory(DATASET_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', subset='validation')
resnet_model.fit(train_rg, validation_data=val_rg, epochs=EPOCHS)
resnet_model.save('models/resnet_model.h5')

print("All DL models trained and saved successfully.")
