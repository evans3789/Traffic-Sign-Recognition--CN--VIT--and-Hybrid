#cnn only
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------- Config ----------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dataset", "Train")
IMG_SIZE = (32, 32)
BATCH_SIZE = 32
EPOCHS = 30
RANDOM_STATE = 42

# ---------- Load dataset ----------
classes_list = sorted(os.listdir(DATA_DIR))
noOfClasses = len(classes_list)
print("Total classes:", noOfClasses)

images, labels = [], []
for idx, cls in enumerate(classes_list):
    class_dir = os.path.join(DATA_DIR, cls)
    for fname in sorted(os.listdir(class_dir)):
        p = os.path.join(class_dir, fname)
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.resize(img, IMG_SIZE)
        images.append(img)
        labels.append(idx)
    print(f"Loaded class {idx}", end=" ")

images = np.array(images)
labels = np.array(labels)
print("\nTotal images:", images.shape[0])

# ---------- Deterministic splits ----------
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, stratify=labels, random_state=RANDOM_STATE
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE
)
print("Shapes - train, val, test:", X_train.shape, X_val.shape, X_test.shape)

# ---------- Preprocessing: grayscale -> equalize -> normalize -> replicate 3 channels ----------
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def equalize(img):
    return cv2.equalizeHist(img.astype(np.uint8))

def preprocess(img):
    g = grayscale(img)
    e = equalize(g)
    n = e.astype(np.float32) / 255.0
    return np.repeat(n[..., np.newaxis], 3, axis=-1)  # replicate to 3 channels

X_train = np.array([preprocess(x) for x in X_train])
X_val   = np.array([preprocess(x) for x in X_val])
X_test  = np.array([preprocess(x) for x in X_test])

y_train_cat = to_categorical(y_train, noOfClasses)
y_val_cat = to_categorical(y_val, noOfClasses)
y_test_cat = to_categorical(y_test, noOfClasses)

# ---------- Augmentation ----------
dataGen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1,
                             zoom_range=0.2, shear_range=0.1, rotation_range=10)
dataGen.fit(X_train)

# ---------- Build CNN ----------
def build_cnn(input_shape=(32,32,3), num_classes=noOfClasses):
    inp = Input(shape=input_shape)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inp)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inp, outputs=out, name='cnn_only')
    return model

model = build_cnn()
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ---------- Train ----------
history = model.fit(
    dataGen.flow(X_train, y_train_cat, batch_size=BATCH_SIZE),
    steps_per_epoch=max(1, len(X_train)//BATCH_SIZE),
    validation_data=(X_val, y_val_cat),
    epochs=EPOCHS,
    shuffle=True
)

# ---------- Evaluate ----------
score = model.evaluate(X_test, y_test_cat, verbose=0)
print("Test loss:", score[0], "Test acc:", score[1])

y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test, y_pred_labels)
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True, fmt='g')
plt.xlabel('Predicted'); plt.ylabel('Truth'); plt.title('CNN Confusion Matrix')
plt.savefig('confusionmatrix_cnn.png', dpi=300, bbox_inches='tight')

model.save('cnn_model.h5')
print("Saved cnn_model.h5")
