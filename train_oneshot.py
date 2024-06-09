import tensorflow as tf
import pandas as pd

# Load the dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    # "D:\\detect_label\\flickr_logos_27_dataset\\classify_by_brand_dataset",
    "D:\\detect_label\\flickr_logos_27_dataset\\test1_data",
    image_size=(224, 224),
)
res_model = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
for layer in res_model.layers:
    layer.trainable = False
x = res_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(512, activation='relu')(x) 
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(256, activation='relu')(x) 
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(128, activation='relu')(x) 
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(64, activation='relu')(x) 
x = tf.keras.layers.Dropout(0.5)(x)
predictions = tf.keras.layers.Dense(2, activation='softmax')(x)
model = tf.keras.models.Model(inputs=res_model.input, outputs=predictions)
model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_ds, epochs=20, verbose=1)
model.save("model.onnx", save_format="onnx")