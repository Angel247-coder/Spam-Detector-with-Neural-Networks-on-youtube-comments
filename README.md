# Detector de Spam con Redes Neuronales — Implementación corregida

Versión depurada del proyecto descrito en la memoria, con dos cambios
de alcance respecto al documento original:

- **Conexión a la YouTube Data API v3** habilitada (el documento la
  excluía explícitamente en "Exclusiones del Producto"; aquí se añade
  como modo de entrada extra en la app Streamlit).
- **Refuerzo anti-overfitting** mediante regularización, dropout, early
  stopping y validación cruzada estratificada.

## Estructura

```
spam_detector/
├── data_prep.py      # Unificación CSVs + ingeniería de features
├── model.py          # Red neuronal + entrenamiento + CV
├── youtube_api.py    # Cliente de YouTube Data API v3
├── train.py          # Script de entrenamiento (CLI)
├── app.py            # Aplicación Streamlit
├── requirements.txt
└── data/             # Coloca aquí los 5 CSV Youtube0X-*.csv
```

## Bugs corregidos respecto a los fragmentos del documento

| # | Fragmento original | Problema | Solución |
|---|---|---|---|
| 1 | Sección 1.2 — uso de `df_unificado['Artista']` | La columna `Artista` no existe en los CSV originales; el código lanzaría `KeyError`. | Se crea al concatenar en `load_and_unify`. |
| 2 | Sección 1.3 — `r'http\|www\|.com'` | `.com` sin escapar: el `.` matchea cualquier carácter → falsos positivos. | Regex anclada con `\b` y `\.com\b`, ampliada a otros TLDs. |
| 3 | Sección 1.3 — orden de cálculo | `ratio_mayusculas` dividía por `longitud_caracteres` antes de calcularla → NaN. | Las longitudes se calculan primero. |
| 4 | Sección 5.5 — `scaler.fit_transform(df[...])` | Ajusta el scaler con TODO el dataset, incluyendo lo que luego será test → fuga de datos. | El `RobustScaler` se ajusta **solo con train** y se aplica a val/test sin `fit`. |
| 5 | Sección 6.5 — `df.dropna(subset=['DATE'])` | El documento decide eliminar 245 filas porque DATE tiene 12.5% de ausentes, pero **DATE no entra al modelo**. Pérdida innecesaria de datos. | Se descarta la **columna** DATE; las filas se conservan. |
| 6 | Sección 4.6 — decisión de eliminar `longitud_palabras` (r=0.91 con `longitud_caracteres`) | Correcta en el documento pero no implementada en el código de features. | Implementada en `FEATURE_COLUMNS`. |

## Medidas anti-overfitting

Dataset pequeño (~2.000 filas) y 5 features → riesgo alto de
memorización. Capas aplicadas:

1. **Arquitectura mínima** (16 → 8 → 1). Más capacidad no mejora con
   tan pocos datos y features.
2. **Regularización L2** (λ = 1e-3) en cada capa densa.
3. **Dropout 0.3** entre capas ocultas.
4. **EarlyStopping** sobre `val_loss` (paciencia 15, restaura mejores
   pesos).
5. **ReduceLROnPlateau** para afinar al final.
6. **`class_weight` balanceado** por si el balance se desvía en algún
   fold (el dataset es 51.4 / 48.6).
7. **K-Fold estratificada (5 folds)** para reportar métricas con
   intervalo (media ± desviación), no un único split que daría
   estimaciones ruidosas con tan pocos datos.
8. **Test holdout 15%** intocable hasta el final.

## Uso

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Coloca los 5 CSV en ./data/

# 3. Entrenar
python train.py --csv-dir ./data

# 4. Lanzar la app
streamlit run app.py
```

Para el modo "URL de YouTube" necesitas una API key gratuita de Google
Cloud Console con la **YouTube Data API v3** habilitada. Se introduce
en la barra lateral de la app, o se exporta como variable de entorno:

```bash
export YOUTUBE_API_KEY="tu_clave_aqui"
```
