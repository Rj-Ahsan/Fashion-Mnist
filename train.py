import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten, Dense, Input, BatchNormalization, Dropout
import matplotlib.pyplot as plt
import numpy as np

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Pixel values now between 0 and 1: min={X_train.min():.2f}, max={X_train.max():.2f}")

from tensorflow.keras.regularizers import l2

model = Sequential()
model.add(Input(shape=(28, 28, 1)))

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu', kernel_regularizer= l2(0.001)))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

from tensorflow.keras.optimizers import Adam

optimizer = Adam(
    learning_rate=0.001
)

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

history = model.fit(X_train, y_train, epochs = 10 , validation_split = 0.2, batch_size = 64, verbose = 1)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Evaluate on training data
train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
print(f"✅ Training Accuracy: {train_accuracy:.4f} ({train_accuracy:.2%})")
print(f"✅ Training Loss: {train_loss:.4f}")
# ============================================
# 1. TEST SET EVALUATION
# ============================================
print("="*60)
print("TEST SET EVALUATION")
print("="*60)

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Test Loss: {test_loss:.4f}")
print(f"✅ Test Accuracy: {test_accuracy:.4f} ({test_accuracy:.2%})")

# Predictions
y_pred_proba = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)

# Calculate additional metrics
from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\n📊 Additional Metrics:")
print(f"  - Precision: {precision:.4f}")
print(f"  - Recall: {recall:.4f}")
print(f"  - F1-Score: {f1:.4f}")

from tensorflow.keras.preprocessing import image

# ============================================
# 1. PREDICT SINGLE IMAGE (From Test Set)
# ============================================

def predict_single_image(model, X_test, y_test, class_names, index):
    """
    Predict and display a single image from test set
    """
    # Get image and true label
    img = X_test[index]
    true_label = y_test[index]

    # Reshape for prediction
    img_input = img.reshape(1, 28, 28, 1)

    # Make prediction
    pred_proba = model.predict(img_input, verbose=0)
    pred_label = np.argmax(pred_proba)
    confidence = np.max(pred_proba)

    # Display results
    print("="*50)
    print(f"🔍 PREDICTION RESULT")
    print("="*50)
    print(f"True Label:    {class_names[true_label]}")
    print(f"Predicted:     {class_names[pred_label]}")
    print(f"Confidence:    {confidence:.2%}")
    print(f"✅ Correct!" if pred_label == true_label else "❌ Incorrect!")

    # Show image
    plt.figure(figsize=(4, 4))
    plt.imshow(img.reshape(28, 28), cmap='gray')
    plt.title(f"True: {class_names[true_label]}\nPred: {class_names[pred_label]} ({confidence:.2%})")
    plt.axis('off')
    plt.show()

    return pred_label, confidence

# Test with a random image from test set
predict_single_image(model, X_test, y_test, class_names, index=100)

from google.colab import files

# ============================================
# 2. UPLOAD AND PREDICT (Colab)
# ============================================

def upload_and_predict(model, class_names):
    """
    Upload an image and get prediction
    """
    print("📤 Please upload an image (28x28 or any size)...")
    uploaded = files.upload()

    for filename in uploaded.keys():
        print(f"\n🔍 Processing: {filename}")

        # Load and preprocess image
        img = image.load_img(filename, target_size=(28, 28), color_mode='grayscale')
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        # Predict
        pred_proba = model.predict(img_array, verbose=0)
        pred_label = np.argmax(pred_proba)
        confidence = np.max(pred_proba)

        # Show all probabilities
        print("\n📊 Prediction Probabilities:")
        print("-"*40)
        for i, name in enumerate(class_names):
            prob = pred_proba[0][i]
            bar = "█" * int(prob * 50)
            print(f"{name:15s}: {prob:.2%} {bar}")

        print("\n" + "="*50)
        print(f"✅ PREDICTION: {class_names[pred_label]}")
        print(f"📊 Confidence: {confidence:.2%}")
        print("="*50)

        # Display image
        plt.figure(figsize=(4, 4))
        plt.imshow(img, cmap='gray')
        plt.title(f"Predicted: {class_names[pred_label]}\nConfidence: {confidence:.2%}")
        plt.axis('off')
        plt.show()

        return class_names[pred_label], confidence

# Run
upload_and_predict(model, class_names)

# ============================================
# 3. PREDICT WITH TOP 3 CLASSES (FIXED)
# ============================================

def predict_top3(model, image_path, class_names):
    """
    Predict and show top 3 most likely classes
    """
    # Load and preprocess
    img = image.load_img(image_path, target_size=(28, 28), color_mode='grayscale')
    img_array = image.img_to_array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Predict
    pred_proba = model.predict(img_array, verbose=0)[0]

    # Get top 3
    top3_indices = np.argsort(pred_proba)[-3:][::-1]
    top3_probs = pred_proba[top3_indices]

    # Print results
    print("\n" + "="*50)
    print("🔍 TOP 3 PREDICTIONS")
    print("="*50)
    for i, (idx, prob) in enumerate(zip(top3_indices, top3_probs), 1):
        print(f"{i}. {class_names[idx]}: {prob:.2%}")
    print("="*50)

    # Display image
    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap='gray')
    plt.title(f"Top: {class_names[top3_indices[0]]}\nConf: {top3_probs[0]:.2%}")
    plt.axis('off')
    plt.show()

    return top3_indices[0], top3_probs[0]

# ============================================
# TEST THE FUNCTION
# ============================================

# Test on a random image from test set
test_idx = 42  # Change this to any index
img_array = X_test[test_idx]

# Save the image temporarily
from PIL import Image
temp_path = 'temp_test_image.png'
Image.fromarray((img_array.reshape(28, 28) * 255).astype('uint8'), mode='L').save(temp_path)

# Predict
pred_label, confidence = predict_top3(model, temp_path, class_names)

print(f"\n✅ Final Prediction: {class_names[pred_label]} ({confidence:.2%})")
print(f"✅ True Label: {class_names[y_test[test_idx]]}")
print(f"✅ Correct!" if pred_label == y_test[test_idx] else "❌ Incorrect!")

