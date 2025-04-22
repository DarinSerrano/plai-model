import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import shutil
import random
import json
import cv2

# Ruta al directorio que CONTIENE las carpetas de las imágenes originales
#original_data_dir = 'tesina/datos/Banana Disease Recognition Dataset/Original Images/Original Images'
original_data_dir = 'tesina/datos/Banana Disease Recognition Dataset/Augmented images/Augmented images'

# Directorio base para los conjuntos divididos
base_dir = 'tesina/banana_disease_split'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# Configuración
image_size = (224, 224) 
batch_size = 64 # tamaño del lote para el generador de datos (32)
epochs = 3 # Numero de épocas para el entrenamiento (10)
num_classes = 7 
class_names_original = ['Banana Black Sigatoka Disease',
                        'Banana Bract Mosaic Virus Disease',
                        'Banana Healthy Leaf',
                        'Banana Insect Pest Disease',
                        'Banana Moko Disease',
                        'Banana Panama Disease',
                        'Banana Yellow Sigatoka Disease']

# Usaremos esta lista para crear las carpetas con nombres consistentes
class_names_for_paths = [name.replace(' ', '_') for name in class_names_original]

class_names_readable = ['Banana Black Sigatoka Disease',
                        'Banana Bract Mosaic Virus Disease',
                        'Banana Healthy Leaf',
                        'Banana Insect Pest Disease',
                        'Banana Moko Disease',
                        'Banana Panama Disease',
                        'Banana Yellow Sigatoka Disease']

def crear_directorios():
    """Crea los directorios para entrenamiento, validación y prueba si no existen."""
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    for class_name in class_names_for_paths:
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(validation_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

def dividir_datos(original_data_dir, train_dir, validation_dir, test_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, random_seed=42):
    """Divide las imágenes en conjuntos de entrenamiento, validación y prueba."""
    random.seed(random_seed)
    for i, class_name in enumerate(class_names_original):
        original_class_dir = os.path.join(original_data_dir, class_name)
        images = [f for f in os.listdir(original_class_dir) if os.path.isfile(os.path.join(original_class_dir, f))]
        random.shuffle(images)

        train_split = int(len(images) * train_ratio)
        val_split = int(len(images) * (train_ratio + val_ratio))

        train_images = images[:train_split]
        val_images = images[train_split:val_split]
        test_images = images[val_split:]

        # Mover imágenes a los directorios correspondientes
        path_class_name = class_names_for_paths[i]
        for img in train_images:
            src = os.path.join(original_class_dir, img)
            dst = os.path.join(train_dir, path_class_name, img)
            shutil.copyfile(src, dst)

        for img in val_images:
            src = os.path.join(original_class_dir, img)
            dst = os.path.join(validation_dir, path_class_name, img)
            shutil.copyfile(src, dst)

        for img in test_images:
            src = os.path.join(original_class_dir, img)
            dst = os.path.join(test_dir, path_class_name, img)
            shutil.copyfile(src, dst)
    print("División de datos completada.")

def cargar_y_preprocesar_datos(train_dir, validation_dir, test_dir, image_size, batch_size):
    """Carga y preprocesa los datos utilizando ImageDataGenerator."""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=42,
        classes=class_names_for_paths # Aseguramos el orden correcto
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        seed=42,
        classes=class_names_for_paths # Aseguramos el orden correcto
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        seed=42,
        classes=class_names_for_paths # Aseguramos el orden correcto
    )

    return train_generator, validation_generator, test_generator

def construir_modelo_transfer_learning(num_classes, image_size):
    """Construye el modelo de Transfer Learning con MobileNetV2."""
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=image_size + (3,))
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def entrenar_modelo(model, train_generator, validation_generator, output_dir, epochs):
    """Entrena el modelo."""
    os.makedirs(output_dir, exist_ok=True)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=5, min_lr=0.00001)
    model_checkpoint = ModelCheckpoint(
        os.path.join(output_dir, 'best_model_{epoch:02d}-{val_accuracy:.4f}.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(output_dir, 'training_log.csv'))

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[early_stopping, reduce_lr, model_checkpoint, csv_logger]
    )
    # Guardar el modelo final
    final_model_path = os.path.join(output_dir, 'final_model.h5')
    model.save(final_model_path)
    print(f"Modelo final guardado en: {final_model_path}")
    #################################################################
    # Generar gráficos de precisión y pérdida
    plt.figure(figsize=(12, 6))

    # Gráfico de precisión
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Precisión de Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Precisión de Validación')
    plt.title('Precisión durante el Entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()

    # Gráfico de pérdida
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida de Validación')
    plt.title('Pérdida durante el Entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()

    # Guardar los gráficos
    metrics_plot_path = os.path.join(output_dir, 'training_metrics.png')
    plt.tight_layout()
    plt.savefig(metrics_plot_path)
    plt.close()
    print(f"Gráficos de métricas de entrenamiento guardados en: {metrics_plot_path}")
    ###############################################################"""
    return history

def evaluar_modelo(model, test_generator, class_names, output_dir):
    """
    Evalúa el modelo con el conjunto de prueba y guarda las métricas.
    Maneja el caso en que algunas clases puedan no estar presentes en el test_generator.
    """
    test_generator.reset()
    y_true = test_generator.classes
    y_pred_probs = model.predict(test_generator)
    y_pred = np.argmax(y_pred_probs, axis=1)

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("Reporte de Clasificación (Conjunto de Prueba):")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    print("\nMatriz de Confusión (Conjunto de Prueba):")
    print(cm)

    metrics = {
        'accuracy': report['accuracy'],
        'weighted_precision': report['weighted avg']['precision'],
        'weighted_recall': report['weighted avg']['recall'],
        'weighted_f1': report['weighted avg']['f1-score'],
        'class_metrics': {}
    }
    for class_name in class_names_readable:
        if class_name in report:
            metrics['class_metrics'][class_name] = report[class_name]
        else:
            metrics['class_metrics'][class_name] = {
                'precision': 0.0,
                'recall': 0.0,
                'f1-score': 0.0,
                'support': 0
            }
    metrics['confusion_matrix'] = cm.tolist()

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    metrics_path = os.path.join(output_dir, f'test_metrics_{current_time}.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\nMétricas de prueba guardadas en: {metrics_path}")

    plt.figure(figsize=(12, 10)) # Agrandamos la figura
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusión (Conjunto de Prueba)')
    plt.xlabel('Clase Predicha')
    plt.ylabel('Clase Real')
    plt.tight_layout() # Ajusta el layout para que no se corten las etiquetas
    cm_path = os.path.join(output_dir, f'confusion_matrix_test_{current_time}.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"Matriz de confusión del conjunto de prueba guardada en: {cm_path}")

    ################################################### generado con copilot en visual studio code
    # Extraer métricas por clase
    class_metrics = {class_name: report[class_name] for class_name in class_names if class_name in report}

    # Crear gráfico de barras para precisión, recall y F1-score
    metrics_df = {
        'Clase': [],
        'Precisión': [],
        'Recall': [],
        'F1-Score': []
    }
    for class_name, metrics in class_metrics.items():
        metrics_df['Clase'].append(class_name)
        metrics_df['Precisión'].append(metrics['precision'])
        metrics_df['Recall'].append(metrics['recall'])
        metrics_df['F1-Score'].append(metrics['f1-score'])

    plt.figure(figsize=(12, 6))
    x = np.arange(len(metrics_df['Clase']))
    width = 0.25

    plt.bar(x - width, metrics_df['Precisión'], width, label='Precisión')
    plt.bar(x, metrics_df['Recall'], width, label='Recall')
    plt.bar(x + width, metrics_df['F1-Score'], width, label='F1-Score')

    plt.xlabel('Clases')
    plt.ylabel('Métricas')
    plt.title('Métricas por Clase (Conjunto de Prueba)')
    plt.xticks(x, metrics_df['Clase'], rotation=45, ha='right')
    plt.legend()

    # Guardar el gráfico
    metrics_bar_path = os.path.join(output_dir, 'class_metrics_bar.png')
    plt.tight_layout()
    plt.savefig(metrics_bar_path)
    plt.close()
    print(f"Gráfico de métricas por clase guardado en: {metrics_bar_path}")
    ##################################################

    return metrics # Añadimos esta línea para devolver el diccionario de métricas

def cargar_modelo_entrenado(model_path):
    """Carga un modelo previamente entrenado."""
    model = tf.keras.models.load_model(model_path)
    return model

def predecir_enfermedad(model, image_path, class_names_original, target_size):
    """Realiza la predicción de la enfermedad en una imagen dada."""
    try:
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_index]

        predicted_class_name = class_names_original[predicted_class_index]

        return predicted_class_name, confidence
    except FileNotFoundError:
        print(f"Error: No se encontró la imagen en la ruta: {image_path}")
        return None, None
    except Exception as e:
        print(f"Ocurrió un error al procesar la imagen: {e}")
        return None, None

def visualize_prediction(image_path, predicted_class, confidence, save_dir):
    """Visualiza la imagen con la predicción y la confianza."""
    try:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title(f"Predicción: {predicted_class}\nConfianza: {confidence * 100:.2f}%")
        plt.axis('off')
        prediction_plot_path = os.path.join(save_dir, f'prediction_{os.path.basename(image_path)}')
        plt.savefig(prediction_plot_path)
        plt.close()
        print(f"Imagen con predicción guardada en: {prediction_plot_path}")
    except Exception as e:
        print(f"Error al visualizar la predicción: {e}")

def visualize_grad_cam(model, image_path, predicted_class_index, layer_name, class_names_original, target_size, save_dir):
    """Genera y visualiza el mapa de calor Grad-CAM."""
    try:
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        grad_model = Model([model.inputs], [model.get_layer(layer_name).output, model.output])

        with tf.GradientTape() as tape:
            conv_output, predictions = grad_model(img_array)
            loss = predictions[:, predicted_class_index]

        grads = tape.gradient(loss, conv_output)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))

        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_output), axis=-1)
        heatmap = np.squeeze(heatmap)

        heatmap = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        original_img = cv2.imread(image_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        resized_original_img = cv2.resize(original_img, (img_array.shape[2], img_array.shape[1]))

        superimposed_img = cv2.addWeighted(resized_original_img, 0.6, heatmap, 0.4, 0)

        plt.imshow(superimposed_img)
        plt.title(f"Grad-CAM para: {class_names_original[predicted_class_index]}")
        plt.axis('off')
        gradcam_path = os.path.join(save_dir, f'gradcam_{class_names_original[predicted_class_index]}_{os.path.basename(image_path)}')
        plt.savefig(gradcam_path)
        plt.close()
        print(f"Mapa de calor Grad-CAM guardado en: {gradcam_path}")

    except Exception as e:
        print(f"Error al generar Grad-CAM: {e}")

def main():
    # 1. Crear directorios para los conjuntos divididos
    crear_directorios()

    # 2. Dividir los datos originales en entrenamiento, validación y prueba
    dividir_datos(original_data_dir, train_dir, validation_dir, test_dir)

    # 3. Cargar y preprocesar los datos utilizando los directorios divididos
    train_generator, validation_generator, test_generator = cargar_y_preprocesar_datos(train_dir, validation_dir, test_dir, image_size, batch_size)

    # 4. Construir el modelo de Transfer Learning
    model = construir_modelo_transfer_learning(num_classes, image_size)

    # 5. Definir el directorio de salida para los resultados del entrenamiento actual
    output_dir = "transfer_learning_results_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(output_dir, exist_ok=True)

    # 6. Entrenar el modelo
    history = entrenar_modelo(model, train_generator, validation_generator, output_dir, epochs)

    # 7. Evaluar el modelo con el conjunto de prueba
    test_metrics = evaluar_modelo(model, test_generator, class_names_readable, output_dir)

    # 8. Ejemplo de cómo cargar el mejor modelo entrenado y predecir una imagen del conjunto de prueba
    best_model_path = os.path.join(output_dir, [f for f in os.listdir(output_dir) if 'best_model' in f][0])
    loaded_model = cargar_modelo_entrenado(best_model_path)

    # Seleccionar una imagen de prueba aleatoria
    ejemplo_clase_prueba_path = random.choice(os.listdir(test_dir))
    ejemplo_imagen_prueba = os.path.join(test_dir, ejemplo_clase_prueba_path, random.choice(os.listdir(os.path.join(test_dir, ejemplo_clase_prueba_path))))

    predicted_class, confidence = predecir_enfermedad(loaded_model, ejemplo_imagen_prueba, class_names_original, image_size)

    if predicted_class:
        print(f"\nPredicción para la imagen de prueba '{ejemplo_imagen_prueba}':")
        print(f"Enfermedad predicha: {predicted_class}")
        print(f"Confianza: {confidence * 100:.2f}%")
        visualize_prediction(ejemplo_imagen_prueba, predicted_class, confidence, output_dir)

        predicted_class_index = class_names_original.index(predicted_class)
        # Nombre de la última capa convolucional en MobileNetV2
        last_conv_layer_name = 'block_16_project_relu'
        visualize_grad_cam(loaded_model, ejemplo_imagen_prueba, predicted_class_index, last_conv_layer_name, class_names_original, image_size, output_dir)

        # Guardar la predicción de prueba en las métricas
        if 'ejemplo_prediccion' not in test_metrics:
            test_metrics['ejemplo_prediccion'] = {}
        test_metrics['ejemplo_prediccion']['ruta_imagen'] = ejemplo_imagen_prueba
        test_metrics['ejemplo_prediccion']['clase_predicha'] = predicted_class
        test_metrics['ejemplo_prediccion']['confianza'] = float(confidence) # Convertir a float para JSON

        metrics_path = os.path.join(output_dir, f'test_metrics_{datetime.now().strftime("%Y%m%d-%H%M%S")}.json')
        with open(metrics_path, 'w') as f:
            json.dump(test_metrics, f, indent=4)
        print(f"\nPredicción de prueba guardada en: {metrics_path}")

    # 9. Predicción en una imagen adicional proporcionada por el usuario
    ruta_imagen_externa = input("\nIngrese la ruta de la imagen que desea predecir: ")
    predicted_class_externa, confidence_externa = predecir_enfermedad(loaded_model, ruta_imagen_externa, class_names_original, image_size)

    if predicted_class_externa:
        print(f"\nPredicción para la imagen '{ruta_imagen_externa}':")
        print(f"Enfermedad predicha: {predicted_class_externa}")
        print(f"Confianza: {confidence_externa * 100:.2f}%")
        visualize_prediction(ruta_imagen_externa, predicted_class_externa, confidence_externa, output_dir)

        predicted_class_index_externa = class_names_original.index(predicted_class_externa)
        last_conv_layer_name = 'block_16_project_relu'
        visualize_grad_cam(loaded_model, ruta_imagen_externa, predicted_class_index_externa, last_conv_layer_name, class_names_original, image_size, output_dir)
    else:
        print("No se pudo realizar la predicción en la imagen externa.")

if __name__ == "__main__":
    main()