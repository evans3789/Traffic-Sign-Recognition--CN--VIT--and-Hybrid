import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from tensorflow.keras import layers, Model, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ============================================================
#                 DATASET PREPARATION
# ============================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
dataUrl = os.path.join(script_dir, "Dataset", "Train")

classes_list = os.listdir(dataUrl)
classes_list = os.listdir(dataUrl)
noOfClasses = len(classes_list)
print("Total Classes:", noOfClasses)

images, labels = [], []
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

# ============================================================
#                 DATA SPLITTING
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

print("Data Shapes:")
print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)

# ============================================================
#                 IMAGE PREPROCESSING
# ============================================================
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

# ViT expects 3 channels
X_train = np.repeat(X_train[..., np.newaxis], 3, -1)
X_val = np.repeat(X_val[..., np.newaxis], 3, -1)
X_test = np.repeat(X_test[..., np.newaxis], 3, -1)

y_train = to_categorical(y_train, noOfClasses)
y_val = to_categorical(y_val, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)

# ============================================================
#                 IMAGE AUGMENTATION
# ============================================================
dataGen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    rotation_range=10
)
dataGen.fit(X_train)

# ============================================================
#                 VISION TRANSFORMER MODEL
# ============================================================
def create_vit_classifier(
    input_shape=(32, 32, 3),
    patch_size=4,
    projection_dim=64,
    transformer_layers=6,
    num_heads=4,
    mlp_head_units=128,
):
    inputs = Input(shape=input_shape)
    # Create patches
    patches = layers.Conv2D(
        filters=projection_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid"
    )(inputs)

    # Flatten patches
    flat_patches = layers.Reshape((-1, projection_dim))(patches)

    # Transformer encoder
    for _ in range(transformer_layers):
        # Layer norm 1 + Multihead Attention
        x1 = layers.LayerNormalization(epsilon=1e-6)(flat_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim
        )(x1, x1)
        x2 = layers.Add()([attention_output, flat_patches])

        # Layer norm 2 + MLP
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = layers.Dense(mlp_head_units, activation="relu")(x3)
        x3 = layers.Dropout(0.1)(x3)
        x3 = layers.Dense(projection_dim)(x3)
        flat_patches = layers.Add()([x3, x2])

    # Classification head
    representation = layers.LayerNormalization(epsilon=1e-6)(flat_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dense(mlp_head_units, activation="relu")(representation)
    outputs = layers.Dense(noOfClasses, activation="softmax")(representation)

    model = Model(inputs=inputs, outputs=outputs)
    return model

vit_model = create_vit_classifier()
vit_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

vit_model.summary()

# ============================================================
#                 TRAINING
# ============================================================
batch_size_val = 32
epochs_val = 40

history = vit_model.fit(
    dataGen.flow(X_train, y_train, batch_size=batch_size_val),
    epochs=epochs_val,
    validation_data=(X_val, y_val),
    steps_per_epoch=len(X_train)//batch_size_val,
    shuffle=True
) 

# ============================================================
#                 PERFORMANCE PLOTS
# ============================================================
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('ViT Loss')
plt.xlabel('epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('ViT Accuracy')
plt.xlabel('epoch')
plt.show()

# ============================================================
#                 EVALUATION
# ============================================================
score = vit_model.evaluate(X_test, y_test, verbose=0)
print('Test Score:', score[0])
print('Test Accuracy:', score[1])

y_pred = vit_model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.savefig('confusionmatrix_vit_only.png', dpi=300, bbox_inches='tight')

# ============================================================
#                 SAVE MODEL
# ============================================================
vit_model.save('vit_model.h5')
print("✅ Vision Transformer model saved successfully as vit_model.h5")
