# hybrid_model.py
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

# ---------------------------
# Data loading + single splits
# ---------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
dataUrl = os.path.join(script_dir, "Dataset", "Train")

classes_list = sorted(os.listdir(dataUrl))
noOfClasses = len(classes_list)
print("Total Classes:", noOfClasses)

images = []
labels = []
for idx, cls in enumerate(classes_list):
    class_dir = os.path.join(dataUrl, cls)
    for fname in os.listdir(class_dir):
        path = os.path.join(class_dir, fname)
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.resize(img, (32, 32))
        images.append(img)
        labels.append(idx)
    print(f"Loaded class {idx}", end=" ")

images = np.array(images)
labels = np.array(labels)
print("\nTotal images loaded:", images.shape[0])

# Single split used for BOTH models
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

print("Shapes - train, val, test:", X_train.shape, X_val.shape, X_test.shape)

# ---------------------------
# Preprocessing (grayscale + equalize + normalize)
# and make 3-channel copies for network input
# ---------------------------
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def equalize(img):
    return cv2.equalizeHist(img.astype(np.uint8))

def preprocess_single(img):
    g = grayscale(img)
    e = equalize(g)
    n = e.astype(np.float32) / 255.0
    # replicate to 3 channels (so both CNN and ViT can accept 3-channel input)
    return np.repeat(n[..., np.newaxis], 3, axis=-1)

X_train = np.array([preprocess_single(x) for x in X_train])
X_val   = np.array([preprocess_single(x) for x in X_val])
X_test  = np.array([preprocess_single(x) for x in X_test])

y_train_cat = to_categorical(y_train, noOfClasses)
y_val_cat = to_categorical(y_val, noOfClasses)
y_test_cat = to_categorical(y_test, noOfClasses)

print("Preprocessed shapes:", X_train.shape, X_val.shape, X_test.shape)

# ---------------------------
# Augmentation
# ---------------------------
dataGen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    rotation_range=10
)
dataGen.fit(X_train)

# ---------------------------
# CNN branch (compact)
# ---------------------------
def make_cnn_branch(input_tensor, feature_dim=256):
    # a compact CNN that outputs a feature vector
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(input_tensor)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(feature_dim, activation='relu', name='cnn_feature')(x)
    return x

# ---------------------------
# ViT branch: explicit patch embedding + positional embedding + class token
# ---------------------------
def make_vit_branch(input_tensor,
                    patch_size=4,
                    projection_dim=64,
                    transformer_layers=6,
                    num_heads=4,
                    mlp_dim=128):
    # patch embedding using Conv2D (kernel=patch_size, stride=patch_size)
    patches = layers.Conv2D(filters=projection_dim,
                            kernel_size=patch_size,
                            strides=patch_size,
                            padding='valid',
                            name='patch_embed')(input_tensor)
    # patches shape: (batch, h_p, w_p, projection_dim)
    h_p = patches.shape[1]
    w_p = patches.shape[2]
    num_patches = int(h_p * w_p)
    # flatten patches to sequence: (batch, num_patches, projection_dim)
    flat_patches = layers.Reshape((num_patches, projection_dim), name='flatten_patches')(patches)

    # create class token (trainable) and positional embeddings
    # class token shape -> (1, projection_dim) as weights
    class_token = tf.Variable(initial_value=tf.zeros(shape=(1, 1, projection_dim), dtype=tf.float32),
                              trainable=True, name='class_token_var')
    # positional embeddings for (num_patches + 1) tokens
    positional_embeddings = tf.Variable(
        initial_value=tf.random.truncated_normal((1, num_patches + 1, projection_dim), stddev=0.02),
        trainable=True, name='pos_embed_var'
    )

    # tile class token across batch and concat
    def add_class_and_pos(inputs):
        batch_size = tf.shape(inputs)[0]
        class_tokens_b = tf.tile(class_token, [batch_size, 1, 1])  # (batch, 1, projection_dim)
        x = tf.concat([class_tokens_b, inputs], axis=1)  # (batch, num_patches+1, projection_dim)
        x = x + positional_embeddings  # broadcast add
        return x

    x = layers.Lambda(add_class_and_pos, name='add_class_pos')(flat_patches)

    attention_scores_to_extract = None
    # Transformer encoder stack
    for i in range(transformer_layers):
        # Layer Norm
        x_norm1 = layers.LayerNormalization(epsilon=1e-6, name=f'trans_{i}_ln1')(x)
        mha_layer = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim // num_heads,
                                             name=f'trans_{i}_mha')

        # calling MHA with return_attention_scores via functional API:
        attn_out_and_scores = mha_layer(x_norm1, x_norm1, return_attention_scores=True)
        attn_out = attn_out_and_scores[0]
        attn_scores = attn_out_and_scores[1]  # shape: (batch, num_heads, query_len, key_len)

        # residual
        x = layers.Add(name=f'trans_{i}_add1')([attn_out, x])

        # Feed-forward (MLP)
        x_norm2 = layers.LayerNormalization(epsilon=1e-6, name=f'trans_{i}_ln2')(x)
        mlp = layers.Dense(mlp_dim, activation='relu', name=f'trans_{i}_mlp_dense1')(x_norm2)
        mlp = layers.Dropout(0.1)(mlp)
        mlp = layers.Dense(projection_dim, name=f'trans_{i}_mlp_dense2')(mlp)
        x = layers.Add(name=f'trans_{i}_add2')([mlp, x])

        # capture attention scores from the last layer
        if i == transformer_layers - 1:
            attention_scores_to_extract = attn_scores

    # At this point x is (batch, num_patches+1, projection_dim)
    # representation = class token output (index 0)
    cls_representation = layers.Lambda(lambda z: z[:, 0], name='cls_rep')(x)  # (batch, projection_dim)

    def extract_cls_to_patches(attn):
        # attn: numpy / TF tensor
        # mean over heads
        attn_mean = tf.reduce_mean(attn, axis=1)  
        cls_to_patches = attn_mean[:, 0, 1:]  
        return cls_to_patches

    attn_vector = layers.Lambda(lambda t: extract_cls_to_patches(t), name='cls_to_patches')(attention_scores_to_extract)
    # Convert the attention vector (num_patches) into a dense representation
    attn_feat = layers.Dense(128, activation='relu', name='attn_dense')(attn_vector)

    #  return cls_representation as another ViT feature 
    vit_feat = layers.Dense(128, activation='relu', name='vit_cls_dense')(cls_representation)

    # final ViT branch feature vector (concat)
    vit_feature_vector = layers.Concatenate(name='vit_concat')([vit_feat, attn_feat])  # e.g. 256 dim

    return vit_feature_vector

# ---------------------------
# Build hybrid model
# ---------------------------
input_shape = (32, 32, 3)
inputs = Input(shape=input_shape, name='input_image')

# both branches share the same input (no duplication of data)
cnn_feat = make_cnn_branch(inputs, feature_dim=256)
vit_feat = make_vit_branch(inputs,
                          patch_size=4,
                          projection_dim=64,
                          transformer_layers=6,
                          num_heads=4,
                          mlp_dim=128)

# combine
combined = layers.Concatenate(name='combined_features')([cnn_feat, vit_feat])
x = layers.Dense(256, activation='relu')(combined)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(noOfClasses, activation='softmax', name='classifier')(x)

hybrid_model = Model(inputs=inputs, outputs=outputs, name='hybrid_cnn_vit')
hybrid_model.compile(optimizer=Adam(learning_rate=0.0005),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
hybrid_model.summary()

# ---------------------------
# Training
# ---------------------------
batch_size = 32
epochs = 30

history = hybrid_model.fit(
    dataGen.flow(X_train, y_train_cat, batch_size=batch_size),
    steps_per_epoch=max(1, len(X_train)//batch_size),
    validation_data=(X_val, y_val_cat),
    epochs=epochs,
    shuffle=True
)

# ---------------------------
# Evaluation + confusion matrix
# ---------------------------
score = hybrid_model.evaluate(X_test, y_test_cat, verbose=0)
print('Test loss:', score[0], ' Test acc:', score[1])

y_pred = hybrid_model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred_labels)
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True, fmt="g")
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title('Hybrid Confusion Matrix')
plt.savefig('confusionmatrix_hybrid.png', dpi=300, bbox_inches='tight')

hybrid_model.save('hybrid_cnn_vit.h5')
print("Saved hybrid_cnn_vit.h5")
