"""
hybrid_model.py
Hybrid model with:
- CNN branch using TRUE 1-channel grayscale input
- ViT branch using 3-channel grayscale (ViT requirement)
- Full attention vector concatenated with class token (variant)

Shared deterministic splits (shared_split_*.npy) preserved.
Outputs:
  hybrid_cnn_vit.h5
  confusionmatrix_hybrid.png
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# ---------- CONFIG ----------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dataset", "Train")
IMG_SIZE = (32,32)
BATCH_SIZE = 32
EPOCHS = 30
RANDOM_STATE = 42
SPLIT_PREFIX = "shared_split"


# ============================================================
#  DATASET SPLITTING
# ============================================================
def create_or_load_splits():
    keys = ["X_train","y_train","X_val","y_val","X_test","y_test"]

    # Load if already saved
    if all(os.path.exists(f"{SPLIT_PREFIX}_{k}.npy") for k in keys):
        print("Loading existing dataset splits...")
        X_train = np.load(f"{SPLIT_PREFIX}_X_train.npy", allow_pickle=True)
        y_train = np.load(f"{SPLIT_PREFIX}_y_train.npy", allow_pickle=True)
        X_val   = np.load(f"{SPLIT_PREFIX}_X_val.npy", allow_pickle=True)
        y_val   = np.load(f"{SPLIT_PREFIX}_y_val.npy", allow_pickle=True)
        X_test  = np.load(f"{SPLIT_PREFIX}_X_test.npy", allow_pickle=True)
        y_test  = np.load(f"{SPLIT_PREFIX}_y_test.npy", allow_pickle=True)
        return X_train, X_val, X_test, y_train, y_val, y_test

    # Otherwise create splits
    classes_list = sorted(os.listdir(DATA_DIR))
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

    images = np.array(images)
    labels = np.array(labels)

    # main train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.20, stratify=labels, random_state=RANDOM_STATE
    )

    # train-val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.20, stratify=y_train, random_state=RANDOM_STATE
    )

    # save splits
    np.save(f"{SPLIT_PREFIX}_X_train.npy", X_train, allow_pickle=True)
    np.save(f"{SPLIT_PREFIX}_y_train.npy", y_train, allow_pickle=True)
    np.save(f"{SPLIT_PREFIX}_X_val.npy", X_val, allow_pickle=True)
    np.save(f"{SPLIT_PREFIX}_y_val.npy", y_val, allow_pickle=True)
    np.save(f"{SPLIT_PREFIX}_X_test.npy", X_test, allow_pickle=True)
    np.save(f"{SPLIT_PREFIX}_y_test.npy", y_test, allow_pickle=True)

    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================
#  PREPROCESSING
# ============================================================
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def equalize(img):
    return cv2.equalizeHist(img.astype(np.uint8))

def preprocess_cnn(img):
    g = grayscale(img)
    e = equalize(g)
    n = e.astype(np.float32) / 255.0
    return n[..., np.newaxis]   # TRUE 1-channel grayscale

def preprocess_vit(img):
    g = grayscale(img)
    e = equalize(g)
    n = e.astype(np.float32) / 255.0
    return np.repeat(n[..., np.newaxis], 3, axis=-1)   # ViT needs 3 channels


# Load data
X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = create_or_load_splits()

X_train_cnn = np.array([preprocess_cnn(x) for x in X_train_raw])
X_val_cnn   = np.array([preprocess_cnn(x) for x in X_val_raw])
X_test_cnn  = np.array([preprocess_cnn(x) for x in X_test_raw])

X_train_vit = np.array([preprocess_vit(x) for x in X_train_raw])
X_val_vit   = np.array([preprocess_vit(x) for x in X_val_raw])
X_test_vit  = np.array([preprocess_vit(x) for x in X_test_raw])

classes_list = sorted(os.listdir(DATA_DIR))
NUM_CLASSES = len(classes_list)

y_train_cat = to_categorical(y_train, NUM_CLASSES)
y_val_cat = to_categorical(y_val, NUM_CLASSES)
y_test_cat = to_categorical(y_test, NUM_CLASSES)


# ============================================================
#  DATA AUGMENTATION
# ============================================================
dataGen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    rotation_range=10
)
dataGen.fit(X_train_vit)


# ============================================================
#  CNN BRANCH — **Single Scale**
# ============================================================
def cnn_branch(inp, feature_dim=128):
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(inp)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)   # 32->16
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(feature_dim, activation='relu', name='cnn_feature')(x)
    return x


# ============================================================
#  VIT BRANCH — Full Attention Vector + Class Token
# ============================================================
def vit_branch(inp, patch_size=4, projection_dim=64,
               transformer_layers=6, num_heads=4, mlp_dim=128):

    # 1. Patch projection using Conv2D
    patches = layers.Conv2D(filters=projection_dim,
                            kernel_size=patch_size,
                            strides=patch_size,
                            padding='valid')(inp)

    # patches → sequence
    h_p = patches.shape[1]
    w_p = patches.shape[2]
    num_patches = int(h_p * w_p)
    flat = layers.Reshape((num_patches, projection_dim))(patches)

    # 2. Class token + positional embedding
    class_token = tf.Variable(
        initial_value=tf.zeros((1,1,projection_dim), dtype=tf.float32),
        trainable=True, name='class_token'
    )

    pos_embed = tf.Variable(
        initial_value=tf.random.truncated_normal(
            (1, num_patches+1, projection_dim), stddev=0.02),
        trainable=True, name='pos_embed'
    )

    def add_token_pos(x):
        bs = tf.shape(x)[0]
        ct = tf.tile(class_token, [bs, 1, 1])
        return tf.concat([ct, x], axis=1) + pos_embed

    x = layers.Lambda(add_token_pos)(flat)

    # 3. Transformer Encoder
    attn_last = None
    for i in range(transformer_layers):
        x_norm1 = layers.LayerNormalization(epsilon=1e-6)(x)
        mha = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim//num_heads)

        att_out, att_scores = mha(
            x_norm1, x_norm1, return_attention_scores=True)

        x = layers.Add()([att_out, x])
        x_norm2 = layers.LayerNormalization(epsilon=1e-6)(x)

        mlp = layers.Dense(mlp_dim, activation='relu')(x_norm2)
        mlp = layers.Dropout(0.1)(mlp)
        mlp = layers.Dense(projection_dim)(mlp)
        x = layers.Add()([mlp, x])

        if i == transformer_layers - 1:
            attn_last = att_scores

    # 4. Extract class token representation
    cls_repr = layers.Lambda(lambda z: z[:, 0])(x)

    # 5. FULL attention vector: class → all patches
    def full_att_vector(att):
        att_mean = tf.reduce_mean(att, axis=1)     # avg across heads
        return att_mean[:, 0, :]                  # full vector: (num_patches+1)

    att_vec = layers.Lambda(full_att_vector)(attn_last)

    # 6. Transform both vectors
    cls_feat = layers.Dense(128, activation='relu')(cls_repr)
    att_feat = layers.Dense(128, activation='relu')(att_vec)

    # 7. Final ViT feature
    vit_feature = layers.Concatenate()([cls_feat, att_feat])
    return vit_feature


#  BUILD HYBRID MODEL
inp_cnn = Input(shape=(32,32,1), name="cnn_gray_input")
inp_vit = Input(shape=(32,32,3), name="vit_rgb_input")

cnn_feat = cnn_branch(inp_cnn)
vit_feat = vit_branch(inp_vit)

combined = layers.Concatenate()([cnn_feat, vit_feat])
x = layers.Dense(256, activation='relu')(combined)
x = layers.Dropout(0.5)(x)
out = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=[inp_cnn, inp_vit], outputs=out, name="hybrid_cnn_vit")
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# ============================================================
#  TRAIN MODEL
# ============================================================
history = model.fit(
    dataGen.flow([X_train_cnn, X_train_vit], y_train_cat, batch_size=BATCH_SIZE),
    steps_per_epoch=max(1, len(X_train_cnn)//BATCH_SIZE),
    validation_data=([X_val_cnn, X_val_vit], y_val_cat),
    epochs=EPOCHS,
    shuffle=True
)


# ============================================================
#  EVALUATION
# ============================================================
score = model.evaluate([X_test_cnn, X_test_vit], y_test_cat, verbose=0)
print("Test loss:", score[0], "Test acc:", score[1])

y_pred = model.predict([X_test_cnn, X_test_vit])
y_pred_labels = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test, y_pred_labels)

plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True, fmt='g')
plt.xlabel('Predicted'); plt.ylabel('Truth')
plt.title('Hybrid Confusion Matrix')
plt.savefig("confusionmatrix_hybrid.png", dpi=300, bbox_inches='tight')

# save model
model.save("hybrid_cnn_vit.h5")

print("Saved hybrid_cnn_vit.h5")
