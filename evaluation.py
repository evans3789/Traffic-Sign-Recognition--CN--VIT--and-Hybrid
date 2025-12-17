import os
import tensorflow as tf
from tensorflow.keras.layers import Lambda

# -----------------------------
# Custom function with inferred shape
# -----------------------------
def add_class_and_pos_with_shape(x):
    """
    Adds class token and positional embedding to input tensor.
    Compatible with Keras deserialization.
    """
    batch_size = tf.shape(x)[0]
    seq_len = tf.shape(x)[1]
    feature_dim = tf.shape(x)[2]

    # Class token
    cls_token = tf.zeros((batch_size, 1, feature_dim))
    x = tf.concat([cls_token, x], axis=1)

    # Positional embedding
    pos_emb = tf.Variable(
        initial_value=tf.random.normal((1, seq_len + 1, feature_dim)),
        trainable=True,
        name="pos_embedding"
    )
    return x + pos_emb

# -----------------------------
# Path configuration
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(BASE_DIR, "Dataset", "Train")
MODEL_PATH = os.path.join(BASE_DIR, "hybrid_cnn_vit.h5")

# -----------------------------
# Parameters
# -----------------------------
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32
LABEL_MODE = "int"
SHUFFLE = False

# -----------------------------
# Sanity checks
# -----------------------------
print(f"Script location : {BASE_DIR}")
print(f"Test directory  : {TEST_DIR}")
print(f"Model path      : {MODEL_PATH}")

if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

print("Test directory contents:", os.listdir(TEST_DIR))

# -----------------------------
# Load test dataset
# -----------------------------
print("\nLoading test dataset...")
test_dataset = tf.keras.utils.image_dataset_from_directory(
    directory=TEST_DIR,
    labels="inferred",
    label_mode=LABEL_MODE,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE
)

class_names = test_dataset.class_names
print(f"Detected {len(class_names)} classes:")
for i, name in enumerate(class_names):
    print(f"  {i}: {name}")

# -----------------------------
# Performance optimization
# -----------------------------
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

# -----------------------------
# Load trained model (with Lambda fix)
# -----------------------------
print("\nLoading trained model...")
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "add_class_and_pos": add_class_and_pos_with_shape,
        "Lambda": Lambda
    },
    safe_mode=False  # required for Lambda layer with Python lambda
)
model.summary()

# -----------------------------
# Evaluate model
# -----------------------------
print("\nEvaluating model on test dataset...")
results = model.evaluate(test_dataset, verbose=1)

# -----------------------------
# Output results
# -----------------------------
test_loss = results[0]
test_accuracy = results[1]

print("\n========== TEST RESULTS ==========")
print(f"Test Loss     : {test_loss:.6f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")
print("=================================")
