import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from tensorflow.keras import Input, Model, layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# ==========================================
#        DATASET PREPARATION
# ==========================================

script_dir = os.path.dirname(os.path.abspath(__file__))
dataUrl = os.path.join(script_dir, "Dataset", "Train")

classes_list = os.listdir(dataUrl)
noOfClasses = len(classes_list)
print("Total Classes:", noOfClasses)

images, labels = [], []
print("Importing Classes...")

for idx, class_name in enumerate(classes_list):
    img_dir = os.path.join(dataUrl, class_name)
    for img_file in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (32, 32))
        images.append(img)
        labels.append(idx)
    print(f"Loaded class {idx}", end=" ")

images = np.array(images)
labels = np.array(labels)

print("\nTotal images loaded:", len(images))

# ==========================================
#        DATA SPLITTING
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

print("Data Shapes:")
print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)

# ==========================================
#        IMAGE PREPROCESSING
# ==========================================
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def equalize(img):
    return cv2.equalizeHist(img)

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255.0
    return img

X_train = np.array(list(map(preprocessing, X_train)))
X_val = np.array(list(map(preprocessing, X_val)))
X_test = np.array(list(map(preprocessing, X_test)))

X_train = X_train.reshape(-1, 32, 32, 1)
X_val = X_val.reshape(-1, 32, 32, 1)
X_test = X_test.reshape(-1, 32, 32, 1)

y_train = to_categorical(y_train, noOfClasses)
y_val = to_categorical(y_val, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)

# ==========================================
#        DATA AUGMENTATION
# ==========================================
dataGen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    rotation_range=10
)
dataGen.fit(X_train)

# ==========================================
#        VISION TRANSFORMER BLOCK
# ==========================================
def transformer_encoder(inputs, num_heads=4, key_dim=64, ff_dim=128, dropout=0.1):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)(x, x)
    x = layers.Add()([x, inputs])
    y = layers.LayerNormalization(epsilon=1e-6)(x)
    y = layers.Dense(ff_dim, activation='relu')(y)
    y = layers.Dropout(dropout)(y)
    y = layers.Dense(inputs.shape[-1])(y)
    out = layers.Add()([x, y])
    return out

# ==========================================
#        HYBRID CNN + ViT MODEL
# ==========================================
def seq_Model():
    input_layer = Input(shape=(32, 32, 1))

    # --- CNN feature extractor ---
    x = Conv2D(60, (5, 5), activation='relu')(input_layer)
    x = Conv2D(60, (5, 5), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(30, (3, 3), activation='relu')(x)
    x = Conv2D(30, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.5)(x)

    # --- Flatten CNN features into token sequence for ViT ---
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = layers.Reshape((16, 16))(x)  # Treat as sequence of 16 tokens
    x = transformer_encoder(x, num_heads=4, key_dim=64, ff_dim=128)

    # --- Classification head ---
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(noOfClasses, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = seq_Model()
print(model.summary())

# ==========================================
#        MODEL TRAINING
# ==========================================
batch_size_val = 30
steps_per_epoch_val = 500
epochs_val = 40

history = model.fit(
    dataGen.flow(X_train, y_train, batch_size=batch_size_val),
    steps_per_epoch=steps_per_epoch_val,
    epochs=epochs_val,
    validation_data=(X_val, y_val),
    shuffle=True
)

# ==========================================
#        PERFORMANCE PLOTS
# ==========================================
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()

# ==========================================
#        EVALUATION AND CONFUSION MATRIX
# ==========================================
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score:', score[0])
print('Test Accuracy:', score[1])

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)



# ==========================================
#        SAVE MODEL
# ==========================================
model.save('hybrid_modell.h5')
print("✅ Model saved successfully as model.h5")
