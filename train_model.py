import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import os

# Veri setinin yolu
# UNUTMA: Colab'da bu yolu Colab klasör yapısına göre değiştireceğiz, şimdilik bu kalsın.
DATA_DIR = 'data/deepfake-faces/deepfake-faces' 

# Model parametreleri
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5 # Hızlı bir başlangıç için 5 deneme
BASE_MODEL = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Veri Ön İşleme ve Artırma
train_datagen = ImageDataGenerator(
    rescale=1./255, # Normalizasyon
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # %20'si doğrulama (validation) için
)

# Eğitim Verisi Yükleyici
# NOTE: deepfake-faces zip'inin içinde deepfake-faces klasörü olduğu varsayıldı.
# Eğer data/deepfake-faces içinde direkt real/fake klasörleri varsa DATA_DIR düzenlenmeli.
train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',  # 0: Fake, 1: Real
    subset='training'
)

# Doğrulama Verisi Yükleyici
validation_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# Model Mimarisi Oluşturma (Transfer Learning)
def create_model(base_model):
    # Base modeli dondur (ağırlıklar değişmesin)
    for layer in base_model.layers:
        layer.trainable = False

    # Yeni katmanları ekle
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x) # İkili sınıflandırma (Real/Fake)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

model = create_model(BASE_MODEL)

# Modeli Derle
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Modeli Eğit
print("Model Eğitimi Başlatılıyor...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    verbose=1
)

# Eğitilmiş Modeli Kaydet
model.save('deepfake_detection_model.h5')
print("\nModel başarıyla deepfake_detection_model.h5 olarak kaydedildi.")