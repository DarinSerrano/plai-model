import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import shutil
import random
import json
import cv2  # Importamos OpenCV para manipulación de imágenes

# Ruta al directorio que CONTIENE las carpetas de las imágenes originales
original_data_dir = 'tesina/datos/Banana Disease Recognition Dataset/Augmented images/Augmented images'
#original_data_dir = 'tesina/datos/Banana Disease Recognition Dataset/Original Images/Original Images'

# Directorio base para los conjuntos divididos
base_dir = 'tesina/banana_disease_split'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# Configuración
image_size = (224, 224)
batch_size = 16  
epochs = 30
num_classes = 8 # Número de clases (incluyendo la nueva clase "No Banana")
class_names_original = ['Banana Black Sigatoka Disease',
                        'Banana Bract Mosaic Virus Disease',
                        'Banana Healthy Leaf',
                        'Banana Insect Pest Disease',
                        'Banana Moko Disease',
                        'Banana Panama Disease',
                        'Banana Yellow Sigatoka Disease',
                        'No Banana']  # Agrega la nueva clase

class_names_for_paths = [name.replace(' ', '_') for name in class_names_original]
class_names_readable = class_names_original

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
        rotation_range=40,    # cambio de 30 a 40 grados 7.4.2
        width_shift_range=0.3, # cambio de 0.25 a 0.3 7.4.2
        height_shift_range=0.3, # cambio de 0.25 a 0.3 7.4.2
        shear_range=0.3, # cambio de 0.2 a 0.3 7.4.2
        zoom_range=0.3, # cambio de 0.2 a 0.3 7.4.2
        horizontal_flip=True,
        vertical_flip=True,  # Añadido volteo vertical
        channel_shift_range=50.0,  # nuevo agregado en 7.4.2 Variación de color
        fill_mode='nearest',
        brightness_range=[0.7, 1.3]  # Añadido ajuste de brillo #cambio de (0.8 a 1.2) a (0.7, 1.3 ) 7.4.2
        # Si quisieras añadir ruido, necesitarías una función de preprocesamiento personalizada aquí.
        # preprocessing_function=add_gaussian_noise
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
        classes=class_names_for_paths
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        seed=42,
        classes=class_names_for_paths
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        seed=42,
        classes=class_names_for_paths
    )

    return train_generator, validation_generator, test_generator

def construir_modelo_transfer_learning(num_classes, image_size):
    """Construye el modelo de Transfer Learning con MobileNetV2."""
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=image_size + (3,))
    for layer in base_model.layers[-20:]:  # Descongelar últimas 20 [-20:] capas 7.4.2
        layer.trainable = True   # Descongelar últimas 20 capas 7.4.2 #congelado debe estar en False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(2048, activation='relu')(x)
    ########################################### Ajuste de Dropout y regularización L1 y L2
    from tensorflow.keras.regularizers import l2
    #from tensorflow.keras.regularizers import l1
    x = Dense(2048, activation='relu', kernel_regularizer=l2(0.0001))(x) #cambios a (0.0001) 7.4.2
    #x = Dense(1024, activation='relu', kernel_regularizer=l1(0.001))(x)
    x = Dropout(0.4)(x) #reducir 7.4.2 (0.4)                       # Dropout para evitar sobreajuste nomalmente 0.5
    x = BatchNormalization()(x) #nuevo ajuste 7.4.1
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.0001))(x) #cambios a (0.0001) 7.4.2 # Nueva capa densa con regularización L2 7.4.1 (0.001)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def entrenar_modelo(model, train_generator, validation_generator, output_dir, epochs):
    """Entrena el modelo y guarda el modelo final."""
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
    csv_logger = CSVLogger(os.path.join(output_dir, 'training_log.csv'))

    model.compile(optimizer=Adam(learning_rate=0.0001), 
                  loss='categorical_crossentropy',
                  metrics=['accuracy']) # Cambiado a learning rate (0.0001) 7.4.2 #normal learning (0.001) 7.4

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

    # Generar y guardar gráficos de precisión y pérdida
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Precisión de Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Precisión de Validación')
    plt.title('Precisión durante el Entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida de Validación')
    plt.title('Pérdida durante el Entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()

    metrics_plot_path = os.path.join(output_dir, 'training_metrics.png')
    plt.tight_layout()
    plt.savefig(metrics_plot_path)
    plt.close()
    print(f"Gráficos de métricas de entrenamiento guardados en: {metrics_plot_path}")

    return history

def evaluar_modelo(model, test_generator, class_names, output_dir):
    """Evalúa el modelo con el conjunto de prueba y guarda las métricas en un solo JSON."""
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

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusión (Conjunto de Prueba)')
    plt.xlabel('Clase Predicha')
    plt.ylabel('Clase Real')
    plt.tight_layout()
    cm_path = os.path.join(output_dir, f'confusion_matrix_test_{current_time}.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"Matriz de confusión del conjunto de prueba guardada en: {cm_path}")

    # Gráfico de barras para métricas por clase
    class_metrics_df = {
        'Clase': [],
        'Precisión': [],
        'Recall': [],
        'F1-Score': []
    }
    for class_name, class_metric in metrics['class_metrics'].items():
        class_metrics_df['Clase'].append(class_name)
        class_metrics_df['Precisión'].append(class_metric['precision'])
        class_metrics_df['Recall'].append(class_metric['recall'])
        class_metrics_df['F1-Score'].append(class_metric['f1-score'])

    x = np.arange(len(class_metrics_df['Clase']))
    width = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, class_metrics_df['Precisión'], width, label='Precisión')
    plt.bar(x, class_metrics_df['Recall'], width, label='Recall')
    plt.bar(x + width, class_metrics_df['F1-Score'], width, label='F1-Score')

    plt.xlabel('Clases')
    plt.ylabel('Métricas')
    plt.title('Métricas por Clase (Conjunto de Prueba)')
    plt.xticks(x, class_metrics_df['Clase'], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    metrics_bar_path = os.path.join(output_dir, 'class_metrics_bar.png')
    plt.savefig(metrics_bar_path)
    plt.close()
    print(f"Gráfico de métricas por clase guardado en: {metrics_bar_path}")

    return metrics

def cargar_modelo_entrenado(model_path):
    """Carga un modelo previamente entrenado."""
    model = tf.keras.models.load_model(model_path)
    return model
##############################################################################
# Función modificada para generar y guardar la comparación con Grad-CAM
def predecir_y_visualizar_comparacion_gradcam(model, image_path, class_names_original, target_size, save_dir, layer_name='block_16_expand', heatmap_intensity=0.6, original_opacity=0.4):
    """Realiza la predicción, genera Grad-CAM y guarda una comparación con la imagen original."""
    try:
        original_img = cv2.imread(image_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_index]
        predicted_class_name = class_names_original[predicted_class_index]
        image_name = os.path.basename(image_path)

        if predicted_class_name == 'No_Banana':
            print(f"\nPredicción para la imagen '{image_path}':")
            print(f"Nombre del archivo: {image_name}")
            print("La imagen no se clasifica como una hoja o parte de planta de banano.")
            return predicted_class_name, confidence, image_name
        else:
            # Generar el mapa de calor Grad-CAM
            grad_model = tf.keras.models.Model(
                [model.inputs],
                [model.get_layer(layer_name).output, model.output]
            )
            with tf.GradientTape() as tape:
                conv_output, predictions_grad = grad_model(img_array)
                loss = predictions_grad[:, predicted_class_index]
            grads = tape.gradient(loss, conv_output)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            heatmap_all_classes = tf.multiply(pooled_grads, conv_output)

            # Seleccionar el heatmap correspondiente a la clase predicha
            heatmap = tf.reduce_sum(heatmap_all_classes[:, :, :, predicted_class_index], axis=-1)

            heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-7)
            heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
            heatmap_uint8 = np.uint8(255 * heatmap)

            # Inspeccionar la forma del heatmap (debería ser (altura, ancho))
            print(f"Forma del heatmap después de seleccionar el canal: {heatmap_uint8.shape}")

            # Colormaps to use
            colormaps = {'jet': cv2.COLORMAP_JET, 'inferno': cv2.COLORMAP_INFERNO}
            comparison_paths = []

            for colormap_name, colormap_value in colormaps.items():
                heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap_value).astype(
                    "float32") / 255.0

                # Superponer el mapa de calor en la imagen original con ajustes de intensidad y opacidad
                superimposed_img = cv2.addWeighted(original_img, original_opacity,
                                                   heatmap_colored, heatmap_intensity, 0)

                # Crear la figura para la comparación
                plt.figure(figsize=(10, 5))

                # Mostrar la imagen original
                plt.subplot(1, 2, 1)
                plt.imshow(original_img)
                plt.title("Imagen Original")
                plt.axis('off')

                # Mostrar la imagen con Grad-CAM
                plt.subplot(1, 2, 2)
                plt.imshow(superimposed_img)
                plt.title(
                    f"Grad-CAM ({colormap_name.capitalize()})\nPredicción: {predicted_class_name} ({confidence * 100:.2f}%)\nbatchsize: {batch_size}\nEpochs: {epochs}")
                plt.axis('off')

                # Guardar la gráfica de comparación
                comparison_path = os.path.join(save_dir, f'comparison_gradcam_{image_name}_{colormap_name}.png')
                plt.tight_layout()
                plt.savefig(comparison_path)
                plt.close()
                print(f"Gráfica de comparación con Grad-CAM ({colormap_name.capitalize()}) guardada en: {comparison_path}")
                comparison_paths.append(comparison_path)

            return predicted_class_name, confidence, image_name
    except FileNotFoundError:
        print(f"Error: No se encontró la imagen en la ruta: {image_path}")
        return None, None, None
    except Exception as e:
        print(f"Ocurrió un error al procesar la imagen: {e}")
        return None, None, None

def main():
    # 1. Crear directorios
    crear_directorios()

    # 2. Dividir datos
    dividir_datos(original_data_dir, train_dir, validation_dir, test_dir)

    # 3. Cargar y preprocesar datos
    train_generator, validation_generator, test_generator = cargar_y_preprocesar_datos(train_dir, validation_dir, test_dir, image_size, batch_size)

    # 4. Construir modelo
    model = construir_modelo_transfer_learning(num_classes, image_size)

    # 5. Definir directorio de salida
    output_dir = "transfer_learning_results_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(output_dir, exist_ok=True)

    # 6. Entrenar modelo
    history = entrenar_modelo(model, train_generator, validation_generator, output_dir, epochs)

    # 7. Evaluar modelo
    test_metrics = evaluar_modelo(model, test_generator, class_names_readable, output_dir)

    # 8. Ejemplo de predicción con el mejor modelo
    best_model_path = os.path.join(output_dir, [f for f in os.listdir(output_dir) if 'best_model' in f][0])
    loaded_model = cargar_modelo_entrenado(best_model_path)

    # Imprimir los nombres de las capas para inspeccionar
    #for i, layer in enumerate(loaded_model.layers):
    #     print(f"{i}: {layer.name}")

    # Imprimir el nombre de la capa de entrada
    #print(f"Nombre de la capa de entrada del modelo: {loaded_model.input.name}")

    ejemplo_clase_prueba_path = random.choice(os.listdir(test_dir))
    ejemplo_imagen_prueba = os.path.join(test_dir, ejemplo_clase_prueba_path, random.choice(os.listdir(os.path.join(test_dir, ejemplo_clase_prueba_path))))

    predicted_class, confidence, image_name = predecir_y_visualizar_comparacion_gradcam(
        loaded_model,
        ejemplo_imagen_prueba,
        class_names_original,
        image_size,
        output_dir,
        layer_name='Conv_1',
        heatmap_intensity=0.8,
        original_opacity=0.3
        # colormap=cv2.COLORMAP_VIRIDIS
    )

    if predicted_class:
        print(f"\nPredicción para la imagen de prueba '{ejemplo_imagen_prueba}':")
        print(f"Nombre del archivo: {image_name}")
        print(f"Enfermedad predicha: {predicted_class}")
        print(f"Confianza: {confidence * 100:.2f}%")

        if 'ejemplo_prediccion' not in test_metrics:
            test_metrics['ejemplo_prediccion'] = {}
        test_metrics['ejemplo_prediccion']['ruta_imagen'] = ejemplo_imagen_prueba
        test_metrics['ejemplo_prediccion']['nombre_archivo'] = image_name
        test_metrics['ejemplo_prediccion']['clase_predicha'] = predicted_class
        test_metrics['ejemplo_prediccion']['confianza'] = float(confidence)

        metrics_path = os.path.join(output_dir, f'test_metrics_{datetime.now().strftime("%Y%m%d-%H%M%S")}.json')
        with open(metrics_path, 'w') as f:
            json.dump(test_metrics, f, indent=4)
        print(f"\nPredicción de prueba y comparación Grad-CAM guardados en: {metrics_path}")

    # 9. Predicción en imagen externa con Grad-CAM (opcional)
    ruta_imagen_externa = input("\nIngrese la ruta de la imagen que desea predecir y visualizar con Grad-CAM (o deje en blanco para omitir): ")
    if ruta_imagen_externa:
        predicted_class_externa, confidence_externa, image_name_externa = predecir_y_visualizar_comparacion_gradcam(loaded_model, ruta_imagen_externa, class_names_original, image_size, output_dir, layer_name='Conv_1')
        if predicted_class_externa:
            print(f"\nPredicción para la imagen '{ruta_imagen_externa}':")
            print(f"Nombre del archivo: {image_name_externa}")
            print(f"Enfermedad predicha: {predicted_class_externa}")
            print(f"Confianza: {confidence_externa * 100:.2f}%")    
        else:
            print("No se pudo realizar la predicción y visualización Grad-CAM en la imagen externa.")
    
    
if __name__ == "__main__":
    main()