import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Funciones
def verificar_tipo_datos(df):
    diccionario = {"nombre_campo": [], "tipo_datos": [], "no_nulos_%": [], "nulos_%": [], "nulos": []}

    for columna in df.columns:
        porcentaje_no_nulos = (df[columna].count() / len(df)) * 100
        diccionario["nombre_campo"].append(columna)
        diccionario["tipo_datos"].append(df[columna].apply(type).unique())
        diccionario["no_nulos_%"].append(round(porcentaje_no_nulos, 2))
        diccionario["nulos_%"].append(round(100-porcentaje_no_nulos, 2))
        diccionario["nulos"].append(df[columna].isnull().sum())

    df_info = pd.DataFrame(diccionario)
        
    return df_info

def boxplot(df, columna):
    '''
    Realiza un boxplot sencillo para una columna determinada.
    '''
    plt.figure(figsize=(15, 2))
    sns.boxplot(data=df, x=df[columna])
    plt.title(f'Boxplot de la columna {columna}')
    plt.show()

def histplot(df, columna, bins=None):
    '''
    Realiza un histplot sencillo para una columna determinada.
    '''
    plt.figure(figsize=(5, 3))
    if bins is None:
        sns.histplot(data=df, x=df[columna])
    else:
        sns.histplot(data=df, x=df[columna], bins=bins)
    plt.title(f'Boxplot de la columna {columna}')
    plt.xticks(range(min(df[columna]), max(df[columna]) + 1, 1))

    plt.show()

def histplot_categoricas(df, nombres_columnas):
    '''
    Crea un histograma de las 4 variables categoricas, especificamente en 2 filas y 2 columnas
    '''
        ## Crear un grid de 2x2 para los histogramas
    fig, axes = plt.subplots(2, 2, figsize=(8, 5))

    # Obtener las columnas categóricas
    # Iterar a través de las columnas y crear histogramas en cada eje
    for i, column in enumerate(nombres_columnas):
        row = i // 2
        col = i % 2
        sns.histplot(data=df, x=column, ax=axes[row, col], multiple="dodge")

    # Ajustar los espacios entre los gráficos
    plt.tight_layout()

    # Mostrar los gráficos
    plt.show()

def countplot(df, columna):
    '''
    Realiza un countplot sencillo para una columna determinada.
    '''
    plt.figure(figsize=(14, 3))

    sns.countplot(data=df, y=df[columna], palette=['red', 'blue'] )

    plt.title(f'Countplot de la columna {columna}')

    # Ajusta los espacios entre subplots y muestra
    plt.tight_layout()
    plt.show()

def countplot_vertical(df, columna):
    '''
    Realiza un countplot sencillo para una columna determinada.
    '''
    plt.figure(figsize=(7, 3))

    sns.countplot(data=df, x=df[columna], palette=['red', 'blue'])

    plt.title(f'Countplot de la columna {columna}')

    # Ajusta los espacios entre subplots y muestra
    plt.tight_layout()
    plt.show()

def bigote_max(columna):
    '''
    Calcula el valor del bigote máximo y la cantidad de valores que se encuentran como valores atípicos.
    '''
    # Cuartiles
    q1 = columna.describe()[4]
    q3 = columna.describe()[6]

    # Valor del vigote
    bigote_max = round(q3 + 1.5*(q3 - q1), 2)
    print(f'El bigote superior de la variable {columna.name} se ubica en:', bigote_max)

    # Cantidad de atípicos
    print(f'Hay {(columna > bigote_max).sum()} valores atípicos en la variable {columna.name}')
    
def valor_mas_frecuente(df, columna):
    '''
    Calcula el valore mas frecuente en una columna, su cantidad y porcentaje respecto del total.
    '''
    # Frecuencias
    moda = df[columna].mode()[0]
    # Cantidad de la mayor frecuencia
    cantidad = (df[columna] == moda).sum()
    # Total de registros
    total = df[columna].count()
    # Porcentaje de la mayor frecuencia
    porcentaje = round(cantidad/total * 100,2)
    print(f'Valor mas frecuente de {columna} es {moda}, con una cantidad de {cantidad} y representa el {porcentaje}%.')

def label_encode_categoricals(df):
    '''
    Genera un nuevo dataframe donde se aplica la codificación de etiquetas (label encoding) a las columnas categóricas
    '''
    # Separar columnas numéricas y categóricas
    numeric_columns = df.select_dtypes(include=['number']).columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    
    # Crear una copia del DataFrame original
    encoded_df = df.copy()
    
    # Aplicar label encoding a las columnas categóricas
    label_encoder = LabelEncoder()
    for col in categorical_columns:
        encoded_df[col] = label_encoder.fit_transform(encoded_df[col])
    
    return encoded_df


    if value <= 10:
        return "menor_10"
    elif value >= 15:
        return "mayor_20"
    else:
        return "entre_10_15"
    
def heatmap_categoricas(df):
    '''
    Realiza un mapa de calor donde se pueden ver los porcentajes de datos para cada clase en cada variable.
    '''
    # Crea un dataframe de resumen
    summary_df = pd.DataFrame()

    # Iterar a través de cada columna en el DataFrame original
    for column in df.columns:
        # Verificar si la columna es de tipo 'object'
        if df[column].dtype == 'object':
            # Obtener las categorías únicas y sus recuentos
            category_counts = df[column].value_counts(normalize=True)
            
            # Crear un DataFrame temporal para esta columna
            temp_df = pd.DataFrame({
                'Categoría': category_counts.index,
                column: category_counts.values * 100
            })
            
            # Establecer la columna 'Categoría' como índice para el DataFrame temporal
            temp_df.set_index('Categoría', inplace=True)
            
            # Unir el DataFrame temporal al resumen general
            summary_df = pd.concat([summary_df, temp_df], axis=1, sort=True)

    # Reemplazar los valores NaN con "null"
    summary_df = summary_df.fillna(-1)
    
    # crea la visualización
    plt.figure(figsize=(14, 10))

    # Crear una escala de colores personalizada
    def custom_cmap(value, alpha=1.0):
        if value == -1:
            return (1, 1, 1, alpha)  # Blanco para el valor -1
        else:
            color = plt.get_cmap('Reds')(value / 100)
            adjusted_color = (color[0] * 0.9, color[1] * 0.5, color[2] * 0.5, alpha)
            return adjusted_color

    num_colors = 100
    cmap = LinearSegmentedColormap.from_list(
        'custom_cmap', [custom_cmap(i) for i in range(-1, num_colors + 1)], num_colors + 2)

    ax = sns.heatmap(summary_df.transpose(), cmap=cmap, vmin=-1, vmax=num_colors, square=False, 
                    annot=True, fmt=".1f", cbar=False, annot_kws={"color": "white"}, 
                    linewidths=0.5, linecolor='grey')
    plt.title("Mapa de Calor de Porcentajes")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.show()
    
def plot_corre_heatmap(corr):
    '''
    Graficar un mapa de calor de las correlación entre las variables analizadas.
    '''
    plt.figure(figsize=(12,10))
    sns.heatmap(corr, cbar = True,  square = False, annot=True, fmt= '.1f'
                ,annot_kws={'size': 15},cmap= 'coolwarm')
    plt.xticks()
    plt.yticks()
    # Arreglamos un pequeño problema de visualización
    b, t = plt.ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t) # update the ylim(bottom, top) values
    plt.show()

def GridSearch_model(model, param_grid, X_train, y_train, X_test, y_test):
    '''
    Entrena un modelo haciendo la técnica de búsqueda de hiperparámetros en grilla y devuelve
    los valores de y predichos para el entrenamiento y testeo, asó como el mejor modelo y sus parámetros.
    '''
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    best_params = grid_search.best_params_
    
    return y_pred_train, y_pred_test, best_model, best_params

def resumen_metricas_PCA(results, y_train_ros, y_test_ros):
    '''
    Permite resumir las mejores métricas para los distintos modelos, tanto para el entrenaiento como el testeo.
    '''
    # Crear una lista de diccionarios para almacenar los resultados
    summary_results = []

    # Resumir métricas de forma más concisa y almacenar en la lista
    for n, models in results.items():
        for model_name, (y_train_pred, y_test_pred, best_model) in models.items():
            report_train = classification_report(y_train_ros, y_train_pred, output_dict=True)
            report_test = classification_report(y_test_ros, y_test_pred, output_dict=True)
            
            summary_results.append({
                'Components': n,
                'Model': model_name,
                'Train_Precision': report_train["macro avg"]["precision"],
                'Train_Recall': report_train["macro avg"]["recall"],
                'Train_F1': report_train["macro avg"]["f1-score"],
                'Test_Precision': report_test["macro avg"]["precision"],
                'Test_Recall': report_test["macro avg"]["recall"],
                'Test_F1': report_test["macro avg"]["f1-score"]
            })

    # Crear un DataFrame a partir de la lista de resultados
    df_summary = pd.DataFrame(summary_results)

    # Reorganizar el DataFrame para tener una columna para cada métrica
    df_summary_metrics = df_summary.pivot(index='Components', columns='Model')

    # Elegir las métricas que deseas visualizar (por ejemplo, precisión, recall, F1-score)
    metrics_to_show = ['Train_Precision', 'Train_Recall', 'Train_F1', 'Test_Precision', 'Test_Recall', 'Test_F1']
    df_metrics_to_show = df_summary_metrics[metrics_to_show]

    # Crear un heatmap
    plt.figure(figsize=(10, 3))
    sns.heatmap(df_metrics_to_show, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Comparación de Métricas por Modelo y Componentes Principales')
    plt.show()
    
    return df_summary

def matriz_confusion_PCA(df_summary, results, y_test, metrica):
    '''
    Realiza la matriz de confusión para el mejor modelo y mejor métrica que se obtuvo para el enfoque de PCA.
    '''
    # df_summary contiene la información de los mejores modelos
    best_model_info = df_summary[df_summary[metrica] == df_summary[metrica].max()].iloc[0]

    # Obtener información del mejor modelo
    best_components = best_model_info['Components']
    best_model_name = best_model_info['Model']

    # Obtener predicciones del mejor modelo
    best_model_preds = results[best_components][best_model_name][1]

    # Obtener matriz de confusión
    conf_matrix = confusion_matrix(y_test, best_model_preds)

    # Crear una visualización de la matriz de confusión
    cm_display = ConfusionMatrixDisplay(conf_matrix)
    cm_display.plot()
    plt.xlabel('Predicciones')
    plt.ylabel('Verdaderos')
    plt.title(f'Confusion Matrix for {best_model_name} (Components: {best_components})')
    plt.show()

def train_and_evaluate_model_estrategia1(estimador, param_grid_clf, X_train_ros, y_train_ros, X_test, y_test):
    '''
    Realiza el entrenamiento para los distintos modelos así como calcular las distintas métricas.
    También calcula la matriz de correlación para el entrenamiento y testeo y devuelve los mejores parámetros.
    '''
    # Crea una instancia de GridSearchCV
    grid_search = GridSearchCV(estimator=estimador, param_grid=param_grid_clf, cv=5)

    # Ajusta el GridSearch al conjunto de datos sobremuestreado
    grid_search.fit(X_train_ros, y_train_ros)

    # Obtiene el mejor modelo y sus hiperparámetros
    best_estimador = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Evalúa el modelo en el conjunto de prueba
    y_pred_test = best_estimador.predict(X_test)
    y_pred_train = best_estimador.predict(X_train_ros)
    
    # Calcula las matrices de confusión
    cm_train = confusion_matrix(y_train_ros, y_pred_train)
    cm_test = confusion_matrix(y_test, y_pred_test)

    # Calcular las métricas que necesitas para entrenamiento y prueba
    train_accuracy = accuracy_score(y_train_ros, y_pred_train)
    train_precision = precision_score(y_train_ros, y_pred_train)
    train_recall = recall_score(y_train_ros, y_pred_train)
    train_f1 = f1_score(y_train_ros, y_pred_train)

    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_precision = precision_score(y_test, y_pred_test)
    test_recall = recall_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test)
    
    # Devolver las métricas en un diccionario
    metrics = {
        'Train Accuracy': train_accuracy,
        'Train Precision': train_precision,
        'Train Recall': train_recall,
        'Train F1': train_f1,
        'Test Accuracy': test_accuracy,
        'Test Precision': test_precision,
        'Test Recall': test_recall,
        'Test F1': test_f1,
    }
        
    return metrics, cm_train, cm_test, best_params

def summary_results_estrtegia1(results):
    '''
    Devuelve un resumen de las métricas a partir de un diccionario que contiene el resumen.
    '''
    # Convertir la lista de resultados en un DataFrame
    data = []

    for model_name, metrics in results:
        metrics['Model'] = model_name
        data.append(metrics)

    df_summary = pd.DataFrame(data)

    # Reordenar las columnas para tener un formato más legible
    columns = ['Model', 'Train Accuracy', 'Train Precision', 'Train Recall', 'Train F1', 'Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1']
    df_summary = df_summary[columns]
    return df_summary

def analisis_sentimiento(review):
    '''
    Realiza un análisis de sentimiento en un texto dado y devuelve un valor numérico que representa el sentimiento.

    Esta función utiliza la librería TextBlob para analizar el sentimiento en un texto dado y
    asigna un valor numérico de acuerdo a la polaridad del sentimiento.

    Parameters:
        review (str): El texto que se va a analizar para determinar su sentimiento.

    Returns:
        int: Un valor numérico que representa el sentimiento del texto:
             - 0 para sentimiento negativo.
             - 1 para sentimiento neutral o no clasificable.
             - 2 para sentimiento positivo.
    '''
    if review is None:
        return 1
    analysis = TextBlob(review)
    polarity = analysis.sentiment.polarity
    if polarity < -0.2:
        return 0  
    elif polarity > 0.2: 
        return 2 
    else:
        return 1 
    
def ejemplos_review_por_sentimiento(reviews, sentiments):
    '''
    Imprime ejemplos de reviews para cada categoría de análisis de sentimiento.

    Esta función recibe dos listas paralelas, `reviews` que contiene los textos de las reviews
    y `sentiments` que contiene los valores de sentimiento correspondientes a cada review.
    
    Parameters:
        reviews (list): Una lista de strings que representan los textos de las reviews.
        sentiments (list): Una lista de enteros que representan los valores de sentimiento
                          asociados a cada review (0, 1, o 2).

    Returns:
        None: La función imprime los ejemplos de reviews para cada categoría de sentimiento.
    '''
    for sentiment_value in range(3):
        print(f"Para la categoría de análisis de sentimiento {sentiment_value} se tienen estos ejemplos de reviews:")
        sentiment_reviews = [review for review, sentiment in zip(reviews, sentiments) if sentiment == sentiment_value]
        
        for i, review in enumerate(sentiment_reviews[:3], start=1):
            print(f"Review {i}: {review}")
        
        print("\n")

def verifica_duplicados_por_columna(df, columna):
    '''
    Verifica y muestra filas duplicadas en un DataFrame basado en una columna específica.

    Esta función toma como entrada un DataFrame y el nombre de una columna específica.
    Luego, identifica las filas duplicadas basadas en el contenido de la columna especificada,
    las filtra y las ordena para una comparación más sencilla.

    Parameters:
        df (pandas.DataFrame): El DataFrame en el que se buscarán filas duplicadas.
        columna (str): El nombre de la columna basada en la cual se verificarán las duplicaciones.

    Returns:
        pandas.DataFrame or str: Un DataFrame que contiene las filas duplicadas filtradas y ordenadas,
        listas para su inspección y comparación, o el mensaje "No hay duplicados" si no se encuentran duplicados.
    '''
    # Se filtran las filas duplicadas
    duplicated_rows = df[df.duplicated(subset=columna, keep=False)]
    if duplicated_rows.empty:
        return "No hay duplicados"
    
    # se ordenan las filas duplicadas para comparar entre sí
    duplicated_rows_sorted = duplicated_rows.sort_values(by=columna)
    return duplicated_rows_sorted

def obtener_anio_release(fecha):
    '''
    Extrae el año de una fecha en formato 'yyyy-mm-dd' y maneja valores nulos.

    Esta función toma como entrada una fecha en formato 'yyyy-mm-dd' y devuelve el año de la fecha si
    el dato es válido. Si la fecha es nula o inconsistente, devuelve 'Dato no disponible'.

    Parameters:
        fecha (str or float or None): La fecha en formato 'yyyy-mm-dd'.

    Returns:
        str: El año de la fecha si es válido, 'Dato no disponible' si es nula o el formato es incorrecto.
    '''
    if pd.notna(fecha):
        if re.match(r'^\d{4}-\d{2}-\d{2}$', fecha):
            return fecha.split('-')[0]
    return 'Dato no disponible'
    
def reemplaza_a_flotante(value):
    '''
    Reemplaza valores no numéricos y nulos en una columna con 0.0.

    Esta función toma un valor como entrada y trata de convertirlo a un número float.
    Si la conversión es exitosa, el valor numérico se mantiene. Si la conversión falla o
    el valor es nulo, se devuelve 0.0 en su lugar.

    Parameters:
        value: El valor que se va a intentar convertir a un número float o nulo.

    Returns:
        float: El valor numérico si la conversión es exitosa o nulo, o 0.0 si la conversión falla.
    '''
    if pd.isna(value):
        return 0.0
    try:
        float_value = float(value)
        return float_value
    except:
        return 0.0
    
def convertir_fecha(cadena_fecha):
    '''
    Convierte una cadena de fecha en un formato específico a otro formato de fecha.
    
    Args:
    cadena_fecha (str): Cadena de fecha en el formato "Month Day, Year" (por ejemplo, "September 1, 2023").
    
    Returns:
    str: Cadena de fecha en el formato "YYYY-MM-DD" o un mensaje de error si la cadena no cumple el formato esperado.
    '''
    match = re.search(r'(\w+\s\d{1,2},\s\d{4})', cadena_fecha)
    if match:
        fecha_str = match.group(1)
        try:
            fecha_dt = pd.to_datetime(fecha_str)
            return fecha_dt.strftime('%Y-%m-%d')
        except:
            return 'Fecha inválida'
    else:
        return 'Formato inválido'

def resumen_cant_porcentaje(df, columna):
    '''
    Cuanta la cantidad de True/False luego calcula el porcentaje.

    Parameters:
    - df (DataFrame): El DataFrame que contiene los datos.
    - columna (str): El nombre de la columna en el DataFrame para la cual se desea generar el resumen.

    Returns:
    DataFrame: Un DataFrame que resume la cantidad y el porcentaje de True/False en la columna especificada.
    '''
    # Cuanta la cantidad de True/False luego calcula el porcentaje
    counts = df[columna].value_counts()
    percentages = round(100 * counts / len(df),2)
    # Crea un dataframe con el resumen
    df_results = pd.DataFrame({
        "Cantidad": counts,
        "Porcentaje": percentages
    })
    return df_results

def bigote_max(columna):
    '''
    Calcula el valor del bigote superior y la cantidad de valores atípicos en una columna.

    Parameters:
    - columna (pandas.Series): La columna de datos para la cual se desea calcular el bigote superior y encontrar valores atípicos.

    Returns:
    None
    '''
    # Cuartiles
    q1 = columna.describe()[4]
    q3 = columna.describe()[6]

    # Valor del vigote
    bigote_max = round(q3 + 1.5*(q3 - q1), 2)
    print(f'El bigote superior de la variable {columna.name} se ubica en:', bigote_max)

    # Cantidad de atípicos
    print(f'Hay {(columna > bigote_max).sum()} valores atípicos en la variable {columna.name}')
    