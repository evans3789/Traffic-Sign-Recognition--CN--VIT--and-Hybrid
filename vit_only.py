
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import tensorflow as tf
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

# ---------- Preprocessing ----------
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
def equalize(img):
    return cv2.equalizeHist(img.astype(np.uint8))
def preprocess(img):
    g = grayscale(img)
    e = equalize(g)
    n = e.astype(np.float32) / 255.0
    return np.repeat(n[..., np.newaxis], 3, axis=-1)

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

# ---------- ViT building blocks ----------
def build_vit(input_shape=(32,32,3),
              patch_size=4,
              projection_dim=64,
              transformer_layers=6,
              num_heads=4,
              mlp_dim=128):
    inputs = Input(shape=input_shape)
    # patch embedding via conv (kernel=patch_size, stride=patch_size)
    patches = layers.Conv2D(filters=projection_dim, kernel_size=patch_size, strides=patch_size, padding='valid')(inputs)
    h_p = patches.shape[1]; w_p = patches.shape[2]
    num_patches = int(h_p * w_p)
    flat = layers.Reshape((num_patches, projection_dim))(patches)

    # class token & positional embeddings (trainable variables)
    class_token = tf.Variable(initial_value=tf.zeros((1,1,projection_dim), dtype=tf.float32), trainable=True, name='class_token')
    pos_embed = tf.Variable(initial_value=tf.random.truncated_normal((1, num_patches+1, projection_dim), stddev=0.02),
                            trainable=True, name='pos_embed')

    def add_token_and_pos(x):
        bs = tf.shape(x)[0]
        ct = tf.tile(class_token, [bs,1,1])
        x2 = tf.concat([ct, x], axis=1)
        return x2 + pos_embed

    x = layers.Lambda(add_token_and_pos)(flat)

    # transformer stack & capture attention from last layer
    attn_scores_last = None
    for i in range(transformer_layers):
        x_norm = layers.LayerNormalization(epsilon=1e-6)(x)
        mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim//num_heads, name=f"mha_{i}")
        out_and_scores = mha(x_norm, x_norm, return_attention_scores=True)
        att_out = out_and_scores[0]
        att_scores = out_and_scores[1]  # (batch, heads, query_len, key_len)
        x = layers.Add()([att_out, x])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        mlp = layers.Dense(mlp_dim, activation='relu')(x)
        mlp = layers.Dropout(0.1)(mlp)
        mlp = layers.Dense(projection_dim)(mlp)
        x = layers.Add()([mlp, x])
        if i == transformer_layers - 1:
            attn_scores_last = att_scores

    # class token representation
    cls_repr = layers.Lambda(lambda z: z[:,0], name='cls_repr')(x)  # (batch, projection_dim)

    # extract class->patch attention vector (mean over heads)
    def cls_to_patches(att):
        att_mean = tf.reduce_mean(att, axis=1)  # (batch, query_len, key_len)
        cls_to_p = att_mean[:,0,1:]             # (batch, num_patches)
        return cls_to_p

    att_vector = layers.Lambda(lambda t: cls_to_patches(t), name='att_vector')(attn_scores_last)
    att_feat = layers.Dense(128, activation='relu', name='att_feat')(att_vector)
    cls_feat = layers.Dense(128, activation='relu', name='cls_feat')(cls_repr)
    vit_feat = layers.Concatenate()([cls_feat, att_feat])  # final vit feature

    # classification head
    out = layers.Dense(noOfClasses, activation='softmax')(vit_feat)
    model = Model(inputs=inputs, outputs=out, name='vit_only')
    return model

vit = build_vit()
vit.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
vit.summary()

# ---------- Train ----------
history = vit.fit(
    dataGen.flow(X_train, y_train_cat, batch_size=BATCH_SIZE),
    steps_per_epoch=max(1, len(X_train)//BATCH_SIZE),
    validation_data=(X_val, y_val_cat),
    epochs=EPOCHS,
    shuffle=True
)

# ---------- Evaluate ----------
score = vit.evaluate(X_test, y_test_cat, verbose=0)
print("Test loss:", score[0], "Test acc:", score[1])

y_pred = vit.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test, y_pred_labels)
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True, fmt='g')
plt.xlabel('Predicted'); plt.ylabel('Truth'); plt.title('ViT Confusion Matrix')
plt.savefig('confusionmatrix_vit.png', dpi=300, bbox_inches='tight')

vit.save('vit_model1.h5')
print("Saved vit_model.h5")
