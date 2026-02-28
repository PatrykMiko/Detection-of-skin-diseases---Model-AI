import pickle
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
import os
from tensorflow.keras.applications import EfficientNetV2L
from tensorflow import data as tf_data
import xgboost as xgb
from mealpy import FloatVar, ESOA
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from tensorflow.keras import mixed_precision

# Enable mixed precision to speed up training and reduce memory usage on compatible GPUs
mixed_precision.set_global_policy('mixed_float16')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# --- 1. SETUP & DATA LOADING ---
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '../..', 'dataset_raw')
if not os.path.exists(data_path):
    print(f"No data folder found at path: {data_path}")
else:
    print(f"Data path set to: {data_path}")

width = 224
height = 224
batch_size = 16

# Load train, validation, and test datasets from directories
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(data_path, 'train'),
    validation_split=None,
    image_size=(height, width),
    batch_size=batch_size,
    label_mode='int'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(data_path, 'val'),
    validation_split=None,
    image_size=(height, width),
    batch_size=batch_size,
    label_mode='int'
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(data_path, 'test'),
    image_size=(height, width),
    batch_size=batch_size,
    label_mode='int',
    shuffle=False # Do not shuffle test data to keep metrics aligned with predictions
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Found {num_classes} classes: {class_names}")

# --- 2. DATA AUGMENTATION & PIPELINE OPTIMIZATION ---
augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]

def data_augmentation(x):
    for layer in augmentation_layers:
        x = layer(x)
    return x

# Apply data augmentation only to the training set
train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))

# Optimize data loading performance by prefetching data into memory
train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
val_ds = val_ds.prefetch(tf_data.AUTOTUNE)
test_ds = test_ds.prefetch(tf_data.AUTOTUNE)

# --- 3. MODEL BUILDING (TRANSFER LEARNING) ---
# Load EfficientNetV2L pre-trained on ImageNet, excluding its top classification layer
base_model = EfficientNetV2L(
    weights='imagenet',
    input_shape=(height, width, 3),
    include_top=False,
)

# Freeze the base model to prevent destroying the pre-trained weights during initial training
base_model.trainable = False
inputs = keras.Input(shape=(height, width, 3))

x = inputs
# Note: training=False keeps BatchNormalization layers in inference mode
x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x) # Prevent overfitting on the dense layer
outputs = keras.layers.Dense(num_classes, dtype='float32')(x) # Cast back to float32 for numerical stability
model = keras.Model(inputs, outputs)

model.summary(show_trainable=True)

# --- 4. PHASE 1: WARM-UP TRAINING ---
# Train only the top custom classification layers
model.compile(
    optimizer=Adam(),
    loss=SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

early_stopping_1 = EarlyStopping(
    monitor='val_loss',
    patience=3,
    verbose=1,
    restore_best_weights=True,
)

epochs = 15
history = model.fit(
    train_ds,
    epochs=epochs,
    validation_data=val_ds,
    callbacks=[early_stopping_1]
)

# --- 5. PHASE 2: FINE-TUNING ---
# Unfreeze the top 180 layers of the base model to adapt them specifically to this dataset
base_model.trainable = True
for layer in base_model.layers[:-180]:
    layer.trainable = False

model.summary(show_trainable=True)

# Recompile the model with a much lower learning rate to make small updates
model.compile(
    optimizer=Adam(1e-5),
    loss=SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

early_stopping_2 = EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=1,
    restore_best_weights=True,
)

total_epochs = epochs + 15
history_fine = model.fit(
    train_ds,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1] + 1, # Resume epoch counting from where Phase 1 left off
    validation_data=val_ds,
    callbacks=[early_stopping_2]
)

# Evaluate the initial Deep Learning model
loss, accuracy = model.evaluate(test_ds)
print(f"Test loss: {loss:.4f}")
print(f"Test accuracy: {accuracy:.4f}")

# (Metrics plotting omitted for brevity, logic remains untouched)
print("Metrics")
y_pred_logits = model.predict(test_ds)
y_pred_probs = tf.nn.softmax(y_pred_logits).numpy()
y_pred_classes = np.argmax(y_pred_logits, axis=1)
y_true = np.concatenate([y for x, y in test_ds], axis=0)
print(classification_report(y_true, y_pred_classes, target_names=class_names))
auc_macro = roc_auc_score(y_true, y_pred_probs, multi_class='ovr', average='macro')
print(f"Macro Average AUC: {auc_macro:.4f}")
auc_weighted = roc_auc_score(y_true, y_pred_probs, multi_class='ovr', average='weighted')
print(f"Weighted Average AUC: {auc_weighted:.4f}")
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

# --- 6. FEATURE EXTRACTION FOR XGBOOST ---
# Create a sub-model that outputs the feature vectors right before the final classification layer
feature_extractor = keras.Model(
    inputs=model.input,
    outputs=model.get_layer("global_average_pooling2d").output
)

def extract_features(dataset):
    all_features = []
    all_labels = []
    for images, labels in dataset:
        features_batch = feature_extractor(images, training=False)
        all_features.append(tf.cast(features_batch, tf.float32).numpy())
        all_labels.append(labels.numpy())
    return np.vstack(all_features), np.hstack(all_labels)

# Reload training data without augmentation to get consistent, static features for XGBoost
train_ds_noaug = tf.keras.utils.image_dataset_from_directory(
    os.path.join(data_path, 'train'),
    image_size=(height, width),
    batch_size=batch_size,
    label_mode='int',
    shuffle=False
).prefetch(tf_data.AUTOTUNE)

# Extract tabular features from the deep learning model
X_train, y_train = extract_features(train_ds_noaug)
X_val, y_val = extract_features(val_ds)
X_test, y_test = extract_features(test_ds)

# --- 7. HYPERPARAMETER OPTIMIZATION (XGBOOST via MEALPY) ---
# Define the objective function for the Egret Swarm Optimization Algorithm (ESOA)
def objective_function(solution):
    max_depth = int(solution[0])
    learning_rate = solution[1]
    n_estimators = int(solution[2])
    subsample = solution[3]
    colsample_bytree = solution[4]

    clf = xgb.XGBClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective='multi:softprob',
        num_class=num_classes,
        tree_method='hist',
        verbosity=0,
        random_state=42
    )

    clf.fit(X_train, y_train)
    predictions = clf.predict(X_val)
    acc = accuracy_score(y_val, predictions)
    return 1.0 - acc # The optimizer minimizes the output, so we return error rate (1 - accuracy)

# Define boundaries for the hyperparameters to search through
problem_dict = {
    "bounds": FloatVar(
        lb=(2, 0.01, 50, 0.5, 0.5), # Lower bounds
        ub=(10, 0.3, 200, 1.0, 1.0), # Upper bounds
        name="xgboost_params"
    ),
    "minmax": "min",
    "obj_func": objective_function
}

# Run the optimization using ESOA
optimizer = ESOA.OriginalESOA(epoch=5, pop_size=15)
g_best = optimizer.solve(problem_dict)

# Retrieve the best parameters found by the optimizer
best_params = {
    'max_depth': int(g_best.solution[0]),
    'learning_rate': g_best.solution[1],
    'n_estimators': int(g_best.solution[2]),
    'subsample': g_best.solution[3],
    'colsample_bytree': g_best.solution[4],
    'objective': 'multi:softprob',
    'num_class': num_classes,
    'tree_method': 'hist'
}

# --- 8. FINAL TRAINING & EVALUATION ---
final_xgb = xgb.XGBClassifier(**best_params)

# Combine train and validation data to maximize data usage for the final tree model
X_full = np.vstack([X_train, X_val])
y_full = np.hstack([y_train, y_val])

final_xgb.fit(X_full, y_full)

y_pred_xgb = final_xgb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_xgb)

print(f"Test accuracy: {accuracy:.4f}")

# (Metrics plotting for XGBoost omitted for brevity, logic remains untouched)
print("Metrics")
print(classification_report(y_test, y_pred_xgb, target_names=class_names))
y_pred_probs_xgb = final_xgb.predict_proba(X_test)
auc_macro = roc_auc_score(y_test, y_pred_probs_xgb, multi_class='ovr', average='macro')
print(f"Macro Average AUC: {auc_macro:.4f}")
auc_weighted = roc_auc_score(y_test, y_pred_probs_xgb, multi_class='ovr', average='weighted')
print(f"Weighted Average AUC: {auc_weighted:.4f}")
cm = confusion_matrix(y_test, y_pred_xgb)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

# --- 9. SAVE ARTIFACTS ---
print("\nSaving model")

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "saved_models")
os.makedirs(output_dir, exist_ok=True)
print(f"\nFiles will be saved in the folder: {output_dir}")

# Save the keras feature extractor model independently
keras_save_path = os.path.join(output_dir, "feature_extractor_effnet_p.keras")
feature_extractor.save(keras_save_path)
print(f"1. Feature extractor saved: {keras_save_path}")

# Save the trained XGBoost model structure and weights
xgb_save_path = os.path.join(output_dir, "classifier_xgboost_p.json")
final_xgb.save_model(xgb_save_path)
print(f"2. XGBoost classifier saved: {xgb_save_path}")

# Save the class names mapping to ensure predictions can be matched to text labels later
class_names_path = os.path.join(output_dir, "class_names_p.pkl")
with open(class_names_path, "wb") as f:
    pickle.dump(class_names, f)
print(f"3. Class names save: {class_names_path}")

print("The model was saved successfully!")