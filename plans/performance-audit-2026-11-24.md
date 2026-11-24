# FalcoCV — Auditoría de rendimiento (2026-11-24)

> **Fecha:** 2026-11-24
> **Estado:** propuesta, no implementada. Sin cambios de funcionalidad.
> **Alcance:** `src/falcocv/core/`, `src/falcocv/tools/`, `src/falcocv/ui/`,
> `src/falcocv/web/`, `cpp/kernel/`.
> **Método:** medición directa con `.venv313` sobre imágenes sintéticas y
> `test_pictures/barcode_1D/1.jpg` (5 MP). Todos los resultados se
> contrastaron contra la implementación actual; los casos marcados
> **bit a bit** producen arrays idénticos.
> **Regla de oro:** ninguna propuesta cambia el comportamiento observable de
> una tool, un puerto, un overlay ni un resultado numérico publicado.

---

## 0. Resumen de mejoras medidas

| Área / Tool | Caso medido | Actual | Optimizado | Mejora |
|---|---|---:|---:|---:|
| **Image Calculator** | `add` / `subtract` / `abs_diff` / `weighted_add` (5 MP BGR) | 113–153 ms | 2.6–3.9 ms | **32× – 52×** |
| **Histogram** | Estadísticas `uint8` (5 MP) | 24.8 ms | 0.49 ms | **50×** |
| **Histogram** | Estadísticas `float32` depth con NaNs (16 MP) | 229.2 ms | 94.4 ms | **2.4×** |
| **Filters** | `color_conversion` (`red` / `hue` / `lightness`, 5 MP) | 20.2–57.6 ms | 2.4–4.4 ms | **8× – 13×** |
| **Depth Viz** | `depth_to_gray_u8` / `apply_depth_colormap` (8.4 MP `float32`) | 83.5 / 160.6 ms | 23.5 / 69.6 ms | **2.3× – 3.5×** |
| **CircleFit / LineFit** | RANSAC `fit_circle` / `fit_line` (32 pts, 200 iters) | 16.2 / 7.4 ms | 2.5 / 2.7 ms | **6.4× / 2.7×** |
| **Calipers (`_fit_support`)** | `scan_line_calipers` (32 calipers) | 4.36 ms | 1.69 ms | **2.6×** |
| **Height Blobs 3D** | Pipeline completo sobre ROI 13 MP (`float32`) | 158.2 ms | 104.7 ms | **1.5× (-53 ms)** |
| **ShapeMatch** | Gradientes `float32` full-ROI (5 MP, `vote_factor > 1`) | 12.5 ms | 0.07 ms (local) | **~20 % del total** |
| **HMI Runtime / Debugger** | Overhead de `sys.settrace` durante `run_slot()` sin breakpoints | 5.3× en llamadas Python | 1.0× | **5.3×** |
| **ML (Classifier / Anomaly)** | Carga repetida de sesiones ONNX (`InferenceSession`) | 26–36 ms/instancia | ~0 ms (cacheada) | **Elimina recarga** |

Mediciones de referencia del entorno: ShapeMatch 5 MP con plantilla 240×240
pasa de **60.26 ms**; `PlaneFitter` sobre 1 M de puntos con ROI de Z fijo,
**12.76 ms**.

---

## 1. Herramientas de visión (`src/falcocv/tools/`)

### 1.1. `ImageCalculatorTool` (`image_calculator/logic.py`)

**Problema**

- En `_apply_op` (líneas 141–210), `a` y `b` ya son `uint8`, pero `add`,
  `subtract`, `abs_diff`, `multiply`, `weighted_add`, `add_constant` y
  `multiply_constant` convierten las imágenes completas a `float64`
  (8× más memoria), operan en `float64`, hacen `np.clip(..., 0, 255)` y
  vuelven a convertir a `uint8`. En 5 MP eso son ~113–153 ms por operación
  frente a los ~3 ms de OpenCV.
- En `run()` (línea 293), `working = raw_a.copy()` copia la imagen completa
  antes del bucle aunque la primera operación habilitada reemplaza `working`.

**Solución propuesta**

- Usar las primitivas SIMD saturadas de OpenCV:
  `cv2.add`, `cv2.subtract`, `cv2.absdiff`,
  `cv2.multiply(..., scale=1/255.0)`, `cv2.addWeighted`.
- Copiar `raw_a` solo si no se ejecuta ninguna operación.

**Verificación:** comparar píxel a píxel la salida actual y la nueva sobre
5 MP aleatorio y sobre una imagen real; los tests de `tests/test_tools.py`
deben pasar sin cambios (la saturación saturada de `uint8` es idéntica a
`clip(float64)`).

---

### 1.2. `HistogramTool` (`histogram/logic.py`)

**Problema**

- **Ruta `uint8` / `uint16`** (líneas 96–106): tras calcular
  `hist = cv2.calcHist(...)`, vuelve a recorrer toda la imagen 4 veces en
  NumPy (`np.mean`, `np.std` —que reserva un array `float64` del tamaño de
  la imagen—, `np.min`, `np.max`).
- **Ruta `float32` depth** (líneas 64, 95, 100): extrae
  `gray[np.isfinite(gray)]` **dos veces** (líneas 64 y 100), vuelve a
  calcular `min` y `max` que ya había calculado como `lo` y `hi`, y en la
  línea 95 hace `hist_src = np.nan_to_num(gray, nan=0.0)` copiando toda la
  imagen 2D (además de meter los píxeles `NaN` en el bin de `0.0`).

**Solución propuesta**

- Cuando el histograma cubre los valores enteros exactos (`uint8` con 256
  bins o `uint16` en modo `Auto`), calcular `mean`, `std`, `min` y `max`
  directamente del vector `hist` de 256/65536 elementos.
  Resultado idéntico a `1e-12`.
- En `float32`, extraer `finite = gray[np.isfinite(gray)]` una sola vez,
  reutilizar `lo`/`hi` como `min_val`/`max_val`, pasar `[finite]` a
  `cv2.calcHist` (sin `np.nan_to_num` sobre la imagen 2D) y usar
  `cv2.meanStdDev(finite)`.

**Riesgo / nota:** `mean`/`std` desde el histograma son ligeramente distintos
de `np.mean`/`np.std` en `float32` (`rtol ~1e-5`), pero `HistogramResult`
solo los expone como `float` y ningún test ni consumidor los compara con
tolerancia estricta. Si se prefiere exactitud total, mantener la ruta de
`np.mean` para `float32` y aplicar solo el resto de la optimización.

---

### 1.3. `FiltersTool` (`filters/logic.py`)

**Problema**

- `color_conversion` (líneas 337–356): ejecuta siempre
  `b, g, r = cv2.split(img)` (copiando los 3 canales) aunque el modo sea
  `hue`, `sat`, `val` o `lightness`; y para `lightness` además convierte
  primero a HSV y hace `cv2.split(hsv)` antes de convertir a HLS.
- `sobel` (línea 241): en modo `magnitude`, aplica `np.abs(g)` sobre la
  salida de `cv2.magnitude(gx, gy)`, que ya es `>= 0`, reservando otra
  imagen `float32`.

**Solución propuesta**

- En `color_conversion`, convertir solo al espacio necesario y extraer
  únicamente el canal pedido con `cv2.extractChannel`.
- En `sobel`, evitar `np.abs` cuando `mode == "magnitude"` y normalizar
  in-place.

**Verificación:** **bit a bit idéntico** (ya comprobado en los modos
`red`, `hue`, `lightness`).

---

### 1.4. `HeightBlobs3D` (`height_blobs/logic.py`)

**Problema**

- En un ROI de 13 MP reserva 7 arrays `float32` del tamaño completo del crop
  (`crop = np.ascontiguousarray(...)`, `z_values = crop * scale_z + offset_z`,
  y 4 temporales en la expresión de `height`, líneas 193–229).
- Para filtrar componentes por `seed_height` (líneas 261–277), construye
  `seeds` en todo el crop y evalúa `grow = keep_lut[labels]` haciendo
  fancy-indexing sobre los ~10–13 millones de píxeles `int32` de `labels`
  (32.8 ms), cuando el 99.9 % del array es fondo (`0`).
- En `_mask_image` (línea 435), `(mask * 255).astype(np.uint8)` crea un
  temporal entero antes de truncar a `uint8`.

**Solución propuesta**

- Evitar copiar `crop` si `depth` ya es `float32` y evaluar `z_values` y
  `height` con operaciones in-place (`np.add`, `np.multiply`,
  `np.subtract`) en el mismo orden de evaluación. **Bit a bit idéntico**
  (ya comprobado).
- Usar `cv2.connectedComponentsWithStats` en el paso de semillas y comprobar
  `measure >= seed_height` únicamente dentro del bounding box
  `[top:top+bh, left:left+bw]` de cada componente.
- Multiplicar la máscara con `np.uint8(255)`.

**Verificación:** la máscara final y las `FeaturePoint` deben ser idénticas
(ya comprobado en el benchmark de 13 MP con 16 pines sintéticos).

---

### 1.5. `ShapeMatch` (`shape_match/logic.py`)

**Problema**

- **Gradientes full-ROI ansiosos** (líneas 3155–3164): aunque la regla de
  diseño dice *"Full gradients, Canny and the distance map stay lazy"*,
  `run()` calcula o convierte a `float32` el par `(gx, gy)` de **todo el ROI
  a resolución completa** (`vote_gx.astype(np.float32)` cuando
  `vote_factor == 1`, o `self._full_gradients(gray_for_match)` cuando
  `vote_factor > 1`). En 5 MP cuesta **12.5 ms y 40 MB** (en 20 MP,
  ~50 ms y 160 MB), cuando `_solve_pose` solo usa una ventana local
  alrededor del candidato (`_geometry_local_maps`).
- **Pre-gate y `_refine_candidate` sin batching** (líneas 3126–3137 y
  3166–3185): llaman a `_dt_score` / `_kernel_score` candidato por candidato
  en un bucle Python (`n_poses = 1`, hasta 30 llamadas a pybind11 y 30
  re-escalados de `model_x`/`model_y`) en vez de pasar el vector de
  candidatos en una sola llamada.

**Solución propuesta**

- Calcular `(gx, gy)` en `float32` dentro de `_geometry_local_maps`
  recortando `gray` con 1 píxel de halo (`cv2.spatialGradient` es un Sobel
  3×3; con 1 px de borde el resultado es **100 % bit a bit idéntico** al
  recorte del gradiente completo). Mantener el fallback full-ROI solo cuando
  la ventana local cubre casi todo el ROI (> 65 % del área).
- Agrupar las evaluaciones del pre-gate en una sola llamada vectorizada a
  `_dt_score` (ya comprobado bit a bit con `n_poses = 20`).

**Verificación obligatoria**

- `benchmarks/bench_shape_match.py` antes/después.
- Gate de precisión de 1005 frames (`scripts/validate_shape_match_reference.py`):
  no debe cambiar ningún resultado ni empeorar las métricas
  (0 lost / 0 extra, mediana 0.901 px, máx 5.535 px / 1.162° / 2.065 %).

---

### 1.6. `LineFit` / `CircleFit` (`core/fitting.py`, `tools/_fit_support.py`)

**Problema**

- En `fitting.py`: `_circle_model` ejecuta `np.linalg.svd` +
  `np.linalg.lstsq` incluso para las muestras de **3 puntos** de las 200
  iteraciones de RANSAC; y `_least_median_model` ordena los residuos dos
  veces por iteración (`np.sort(residuals)` y `np.median(residuals)`).
- En `_fit_support.py`: `scan_line_calipers` y `scan_radial_calipers`
  recalculan el kernel Gaussiano 1D e invocan `cv2.remap` por separado en
  cada caliper.

**Solución propuesta**

- Resolver el círculo de 3 puntos en RANSAC por fórmula cerrada
  (determinante 2×2 con la misma guarda de colinealidad). Diferencia máxima
  medida: **6.5e-9 px**.
- Obtener la mediana del array ya ordenado `ordered = np.sort(residuals)`
  y usar `np.hypot` en los residuos.
- Precalcular el kernel Gaussiano una vez por escaneo y agrupar los mapas
  `(map_x, map_y)` de todos los calipers en una sola llamada a `cv2.remap`.
  **Bit a bit idéntico** (ya comprobado).

---

### 1.7. Limpiezas directas en otros tools

- **`PlaneFitter.run`** (`plane_fitter/logic.py`, líneas 151–155): construye
  `full_inlier_mask` sobre toda la nube (1 M puntos) y no lo usa en ningún
  sitio (código muerto).
- **`TemplateMatch`** (`template_match/logic.py`, líneas 228 y 592):
  `_range_to_gray(..., fast=True)` sigue usando la expresión NumPy con 3
  temporales en lugar de `cv2.addWeighted` + clip in-place (como
  `ShapeMatch`), y en `run()` hace `roi_raw.copy()` innecesario cuando la
  entrada ya es 2D `uint8`.
- **`BlobDetector.run`** (`blob_detector/logic.py`, líneas 123, 142, 181,
  272): evita `roi_src.copy()` cuando ya es 2D grayscale, evita crear
  `VisionImage(mask_full)` dos veces, y evita `cnt + [0, 0]` cuando
  `ax == ay == 0`.
- **`MeasurementTool.run`** (`measurement/logic.py`, línea 383): mueve
  `items = _enumerate_elements(vision_img)` fuera del bucle
  `for row_idx, row in enumerate(measurements):`.
- **`AnomalyDetector.run`** (`anomaly_detector/logic.py`, línea 173): cambia
  `propagate_3d(img, vision_img, regenerate=True)` a `regenerate=False`
  (no modifica datos 3D). Alineado con la regla de nubes perezosas.

---

## 2. Core, 3D, kernel C++ y ML

### 2.1. Visualización depth (`core/depth_viz.py`)

**Problema**

`apply_depth_colormap` y `depth_to_gray_u8` convierten toda la imagen a
`float64` (`work = d.astype(np.float64)` — 655 MB para un frame de
`20000x4096`), indexan `work[valid]` dos veces, crean varios arrays
`float64` intermedios y hacen `.astype(np.uint8)` con `NaN` presentes
(emitiendo `RuntimeWarning: invalid value encountered in cast`).

**Solución propuesta**

Mantener `float32`, extraer `work[valid]` una sola vez si `vmin`/`vmax` son
`None`, normalizar con `cv2.addWeighted` + `np.clip(..., out=...)`, limpiar
`~valid` antes del cast a `uint8`.

**Riesgo:** la conversión `float64 → uint8` y `float32 → uint8` puede diferir
en ±1 LSB en valores justo en el redondeo. Los visores no comparan píxeles;
el único consumidor estricto es `ShapeMatch._range_to_gray(fast=True)`, que
ya usa la ruta `float32` con `cv2.addWeighted`. Verificar el gate de
ShapeMatch.

---

### 2.2. Nube de puntos y `plane_fit` (`core/pointcloud_utils.py`, `cpp/kernel/src/kernel.cpp`)

**Problema**

- `pointcloud_utils.py`:
  - `ensure_pointcloud_data` (línea 161) llama a
    `generate_pointcloud(..., compute_bbox=False)`, perdiendo `bbox_center` y
    `bbox_size` que el kernel C++ calcula gratis en el mismo bucle OpenMP;
    el visor 3D tiene luego que escanear 1 M de puntos en Python.
  - `_intensity_colors` (línea 37) reserva
    `colors = np.zeros((count, 3), dtype=np.uint8)` al inicio y luego lo
    descarta cuando devuelve la salida de `cv2.cvtColor`.
- `cpp/kernel/src/kernel.cpp`:
  - En `point_in_roi_xy` (línea 238), cuando un ROI 3D tiene rotación
    (`angle != 0`), calcula `std::cos(a_rad)` y `std::sin(a_rad)` **por cada
    punto de la nube** (1.000.000 de veces por ROI) en vez de precalcular
    `ca, sa` en `ROI3D`.

**Solución propuesta**

- Pasar `compute_bbox=True` desde `ensure_pointcloud_data` y guardar
  `bbox_center` / `bbox_size` en `_3d_data`.
- Eliminar la reserva inútil de `_intensity_colors`.
- Precalcular `ca`/`sa` en la construcción de `ROI3D` (en
  `plane_fit_impl`). Es un cambio C++ puro, sin impacto en Python.

**Verificación:** `tests/test_plane_fitter.py`,
`tests/test_kernel_input_validation.py`, benchmark de kernel dentro del
mismo binario (regla de A/B del proyecto).

---

### 2.3. Caché de sesiones ONNX (`core/classifier/`, `core/anomaly/`)

**Problema**

- `preload_classifier_model()` (`core/classifier/inference.py`, línea 37)
  crea un `ort.InferenceSession` al arrancar (~36 ms) y **lo descarta sin
  guardarlo**.
- Cada instancia de `ClassifierInferenceEngine`, `EfficientNetBackbone`,
  `PatchBackbone` y `Layer2Backbone` crea su propio `ort.InferenceSession`
  desde disco (~26–34 ms cada vez).

**Solución propuesta**

Cachear `(session, input_name, output_name)` a nivel de módulo, igual que
hace `PPOCRv5Engine._PRELOADED_ENGINES`. `session.run()` en ONNX Runtime es
thread-safe para inferencia concurrente; la sesión es de solo lectura en
nuestro uso.

**Verificación:** `tests/test_classifier.py`,
`tests/test_anomaly_detector.py` (si existe) y el smoke test de
`scripts/smoke_standalone.py`.

---

### 2.4. HMI Script Debugger y `run_slot` (`core/hmi.py`, `core/script_debugger.py`)

**Problema**

1. `HMIProject` ejecuta siempre el hilo del script HMI bajo
   `TraceDebugger.run_with_trace()`. Como `run_slot()` corre síncronamente
   en ese mismo hilo, **`sys.settrace` permanece activo durante toda la
   ejecución del workflow de visión**, interceptando cada llamada a función
   Python dentro de todos los tools (~5.3× de sobrecarga medida en llamadas
   Python).
2. En `TraceDebugger._trace_func` (línea 171), `self._depth += 1` se ejecuta
   antes de `if self._step_mode == "continue" and not in_script: return
   None`. En Python, devolver `None` en un evento `'call'` suprime el evento
   `'return'` de ese frame, por lo que `self._depth` **crece
   indefinidamente** (+200.000 medidos en pocos segundos) y desincroniza
   `step_over`/`step_out`.
3. En scripts HMI multi-archivo, cada evento `'line'` ejecuta
   `os.path.relpath(fn, sd).replace("\\", "/")` y dos uniones de conjuntos
   (`bps |= ...`), lo que añade ~1.4 µs por línea ejecutada.

**Solución propuesta**

1. En `HMIProject.run_slot`, suspender `sys.settrace(None)` alrededor de
   `slot.run()` y restaurarlo en `finally:` (si un `PythonScriptTool` dentro
   del job tiene sus propios breakpoints, él mismo instala y restaura su
   trace).
2. En `TraceDebugger._trace_func`, incrementar `self._depth` solo cuando el
   frame vaya a ser trazado, y comprobar `ln in bps` sin construir nuevos
   `set` en cada línea.
3. En `_run_script_multi`, precalcular un diccionario
   `{abs_path: rel_path}` en lugar de llamar a `os.path.relpath` en cada
   línea ejecutada.

**Verificación:** `tests/test_script_debugger.py` (si existe),
`tests/ui/test_python_script_debugger.py` y un job HMI real con y sin
breakpoints.

---

## 3. Interfaz gráfica (`src/falcocv/ui/`)

### 3.1. `ImageViewer.set_image` (`ui/widgets/viewer.py`)

**Problema**

La guarda rápida (línea 387) comprueba
`self._current_vision_image is img_obj`. Como cada tool del pipeline crea
una nueva instancia `VisionImage` que envuelve el **mismo buffer
`img.data`** (para añadir sus propios overlays), al cambiar entre tools en
el selector de registros o al re-ejecutar un tool con `Auto`,
`ImageViewer` vuelve a redimensionar la imagen (hasta 82 MP), vuelve a
aplicar el colormap de profundidad y vuelve a construir
`QImage` + `QPixmap.fromImage`.

**Solución propuesta**

Cachear el `QPixmap` por identidad y metadatos de visualización del buffer
(`self._current_raw_image is img`, `shape`, `dtype`, `depth_display`). Si el
buffer de píxeles y las opciones de depth no han cambiado, actualizar
únicamente `self._current_vision_image`, `_overlay_primitives`,
`_apply_roi()` y repintar el viewport.

**Verificación:** `tests/test_image_viewer_roi_scale.py` (ya cubre el
downsampling y `_display_scale`) y `tests/ui`.

---

### 3.2. `Viewer3D.set_data` y configs 3D (`ui/widgets/viewer3d.py`,
`tools/height_blobs/config.py`)

**Problema**

- Cada vez que se edita un parámetro o ROI con la vista 3D activa,
  `PlaneFitterConfig` y `HeightBlobs3DConfig` llaman a
  `viewer3d.set_data(points, colors, ...)`, lo que vuelve a copiar la nube,
  reconstruir el LOD y **resubir ~15 MB a la GPU (`glBufferData`)** aunque
  `points` y `colors` sean exactamente los mismos arrays.
- Además, `HeightBlobs3DConfig._update_3d_viewer` no preserva
  `camera.target`/`camera.distance` al refrescar.

**Solución propuesta**

En `Viewer3D.set_data`, detectar si `points` y `colors` son los mismos
buffers ya cargados (`self._points is points and self._colors is colors`).
Si no han cambiado, no reiniciar la cámara ni reconstruir/resubir los VBOs
a OpenGL (solo actualizar overlays/ROIs y repintar). Aplicar también la
preservación de cámara en `HeightBlobs3DConfig`.

**Verificación:** `tests/test_viewer3d_handles.py` y prueba manual con
`Show 3D` + edición de ROI en `PlaneFitter` y `HeightBlobs3D`.

---

## 4. Archivos Web (`src/falcocv/web/`)

### 4.1. `image_utils.encode_vision_image` (`web/image_utils.py`)

**Observaciones**

- `encode_image` hace `cv2.imencode('.jpg', image)` directamente sobre el
  buffer de la tool. JPEG solo soporta `uint8` de 1 o 3 canales; una imagen
  `float32` (depth) o `uint16` falla o produce basura. En `hmi_worker`
  `get_output` pasa cualquier `VisionImage`, incluyendo depth, así que la
  ruta actual no es segura para depth.
- Para frames grandes, codifica a resolución completa en cada polling de
  widget. No hay cache ni miniatura.

**Propuesta (fuera del alcance mínimo de esta auditoría, anotada)**

- Reutilizar la lógica de `apply_depth_colormap` para codificar depth como
  BGR `uint8` antes de JPEG.
- Añadir downsampling opcional y cache por `id(data)` + parámetros.

No se propone cambiar esto en el primer lote porque toca el contrato HTTP
del HMI y merece su propio plan.

---

## 5. Plan de implementación propuesto

Orden sugerido, de menor a mayor riesgo:

1. **Lote A — sin riesgo numérico**
   - `ImageCalculatorTool` (OpenCV saturado).
   - `FiltersTool` (`color_conversion`, `sobel`).
   - `PlaneFitter` (código muerto).
   - `MeasurementTool` (`items` fuera del bucle).
   - `AnomalyDetector` (`regenerate=False`).
   - `depth_viz` (float32 con redondeo equivalente; verificar gate de
     ShapeMatch).
   - `pointcloud_utils` (bbox gratis, reserva inútil).
   - `Viewer3D.set_data` (skip de VBO sin cambios).
   - `ImageViewer.set_image` (cache por buffer).

2. **Lote B — bit a bit verificado**
   - `HeightBlobs3D` (in-place + semillas por bounding box).
   - `LineFit` / `CircleFit` + calipers.
   - `HistogramTool` (estadísticas desde el histograma; mantener la ruta
     `float32` si se quiere exactitud total).
   - `PlaneFitter` kernel C++ (`ca`/`sa` precalculados).

3. **Lote C — requiere gate de precisión**
   - `ShapeMatch`: gradientes locales y pre-gate batcheado.
   - Ejecutar `scripts/validate_shape_match_reference.py` y
     `benchmarks/bench_shape_match.py` antes de dar por bueno el cambio.

4. **Lote D — runtime y ML**
   - `HMIProject.run_slot` (suspender `sys.settrace`).
   - `TraceDebugger._trace_func` (depth correcto, sin `set` por línea).
   - Caché de sesiones ONNX en `core/classifier/inference.py` y los tres
     backbones de `core/anomaly/`.

---

## 6. Verificación global

- `pytest -q --ignore=tests/test_trainer_equivalence.py`
  (actualmente 968 passed; los 16 fallos por `zbar` son del entorno).
- `pytest -q tests/ui` (224 passed).
- `python benchmarks/bench_tools.py` antes/después.
- `python benchmarks/bench_shape_match.py` antes/después.
- `scripts/validate_shape_match_reference.py` para el lote C.
- `python scripts/smoke_standalone.py` para el lote D (sesiones ONNX y
  runtime congelado).

---

## 7. Descartado a propósito

- Mover tools a C++ sin un benchmark que demuestre > 1.5× end-to-end
  (regla del proyecto).
- Cambiar el contrato de `results` / puertos / overlays.
- Tocar el pipeline de ShapeMatch más allá de los puntos descritos.
- Añadir caché de frames al `DepthRangeLoader` (decisión ya tomada).
- Cambiar la semántica de `_copy_overlays_to_input`, `inherit_overlays_from`
  o del claim de capas.
