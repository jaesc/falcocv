# FalcoCV — Plan de arquitectura y producto (v2)

> **Fecha:** 2026-10-27
> **Estado:** vivo. Sustituye a los planes eliminados `falcocv-plan-maestro.md` y `tools_cpp_porting.md` (disponibles en el historial de git).
> **Base:** repositorio `falcocv`, rama `main`.
> **Commits de referencia:** `22c5ce6` (fix de config/help), `ede8a88` (red de tests), `945cc3f` (suelo de cobertura).
> **Propósito:** hoja de ruta autocontenida para las próximas fases, con las decisiones ya tomadas y lo que se descarta.

---

## 0. Cómo usar este documento

1. Este plan manda sobre el plan maestro antiguo. El plan maestro es una **fuente de ideas**, no una autoridad.
2. Las decisiones de la sección 2 están cerradas. Si se cambia una, se anota aquí y se actualiza la tabla de estado.
3. Cada fase tiene entregables, criterios de aceptación y tests. No se pasa a la siguiente sin cumplirlos.
4. Los benchmarks se guardan en `benchmarks/` y las decisiones técnicas en el
   informe de decisión de cada fase, no solo en el chat.

---

## 1. Estado actual

### 1.1 Producto

FalcoCV es un entorno de visión industrial con:

- editor de workflows y jobs `.cvj`;
- tools de filtros, histograma, calculadora, blob, bordes, pattern matching, códigos, OCR, clasificador y anomalías;
- profundidad, nubes de puntos, plane fitter y visor 3D;
- fixtures y overlays vectoriales;
- scripting con debugger;
- diseñador HMI, servidor FastAPI, dashboard y workers;
- adquisición GenTL/GigE.

### 1.2 Arquitectura

```
src/falcocv/
  core/        workflow, VisionImage, overlays, ROI, modelos, 3D
  tools/       lógica + configuración PyQt6 por tool
  ui/          widgets, viewers, ROI editors, HMI designer
  web/         FastAPI, workers HMI, dashboard
  main.py      entrada del editor
  cli.py       `falcocv`, `self-test`, `download-models`
cpp/kernel/    _falco_kernel (pybind11, sin OpenCV):
               generate_pointcloud, plane_fit
```

Regla: `core/` no importa PyQt6 ni UI a nivel de módulo. El kernel expone funciones dict→dict/ndarray, nunca `BaseTool`, `VisionImage` ni overlays.

### 1.3 Calidad actual

- **212 tests pasan**, 3 skipped, 3 xfailed (~25 s).
- Tests de UI offscreen (51), integración headless (44), web (7), contrato de tools, jobs, barcode, depth, empaquetado.
- `falcocv self-test` valida todos los configs, ayuda, pipeline sintético y round-trip de job.
- `scripts/check_wheel.py` y `scripts/wheel_smoke.py` verifican la wheel instalada.
- CI: `full` (Windows/Linux, Python 3.13), `compat` (3.10), `package` (wheel + self-test).
- Cobertura: branch, sin `tools/*/config.py`, suelo del 35 % en CI. Es un suelo anti-regresión, no un objetivo.

### 1.4 Huecos reales

1. **Localización robusta**: `PatternMatcher` es correlación (`matchTemplate`) con barrido de ángulo/escala; no es shape matching por gradiente.
2. **Metrología**: `EdgeDetection` ya es un caliper 1D afín con subpíxel, pero faltan `LineFit`/`CircleFit`, cadena de fixture completa, calibración y incertidumbre.
3. **Calibración**: solo homografía 2D; faltan intrínsecos, distorsión y hand-eye.
4. **Inspección**: presencia/ausencia y multi-instancia dependen de un matcher robusto; clasificador/anomalías/OCR necesitan flujo de validación y umbrales.
5. **Producto**: no hay historial/trazabilidad, ni PLC/OPC UA/I/O, ni despliegue industrial sin instalar Python.
6. **Validación**: no hay banco de verdad sintética ni métricas de pose/repetibilidad.

---

## 2. Decisiones cerradas

| ID | Decisión |
|---|---|
| D1 | **Híbrido**: Python/OpenCV/NumPy para core y tools; C++ solo en `cpp/kernel` para hot paths medidos (>1,5× end-to-end). |
| D2 | **No** reescritura C++20 del motor, `cvcore::Image`, Eigen/oneTBB/vcpkg, runtime C++ headless ni `.shm`. |
| D3 | Plataforma: **Windows + Linux**. Sin macOS en v1. |
| D4 | **Headless con Python**. No hay runtime C++ headless. |
| D5 | El **orden de la lista de tools es el contrato de ejecución**. Lo decide el autor del pipeline. No hay orden topológico automático, ni auto-order, ni validación de orden/ciclos, ni avisos que cuestionen el diseño. |
| D6 | Metrología e inspección son **igual de importantes**; comparten ShapeMatch y el contrato de resultados. |
| D7 | El contrato de resultados es **aditivo**: `value`, `unit`, `tolerance`, `verdict`, `score`, `uncertainty`. No rompe `.cvj` ni `ToolResults`. |
| D8 | Licencia **GPLv3**; sin split LGPL/core. |
| D9 | Distribución v1: wheel/sdist + venv. Servidor/HMI en contenedor. Python embebido/portable se valora después de v1 si el cliente no quiere instalar Python. |
| D10 | No PyInstaller/Inno por defecto (regla actual de `AGENTS.md`). Si se necesita un instalador, se evalúa Python embebido oficial + launcher, o se cambia la regla explícitamente. |
| D11 | Cualquier nueva función C++ debe vivir en `cpp/kernel`, no depender de OpenCV y traer benchmark + informe de decisión; debe superar **1,5× end-to-end** frente a la versión Python. |
| D12 | La corrección del pipeline es responsabilidad del autor. El programa ejecuta y reporta errores de runtime; no opina sobre el diseño. |

---

## 3. Qué se conserva y qué se descarta del plan maestro

### Se conserva

- Shape matching por dirección de gradiente como dirección técnica.
- Caliper, ajuste de recta/círculo, fixture y calibración.
- Verdad sintética, métricas de pose, repetibilidad y benchmarks.
- Procedencia limpia y nomenclatura propia.
- Modelos descargados fuera del wheel.

### Se descarta o se adapta

| Plan maestro | Decisión |
|---|---|
| Core C++20 con Python como plano de control | Descartado. Python es el motor. |
| `cvcore::Image` propio, vcpkg, Eigen, oneTBB | Descartado. NumPy/OpenCV. |
| F1 “separar core de Python” | No aplica: el core ya es Python y el kernel no tiene pybind11 en headers públicos. |
| Runtime C++ headless | Descartado. `falcocv run` con Python. |
| Formato `.shm` | Descartado. Arrays comprimidos dentro del `.cvj`. |
| OCR propio con CRNN/CTC | Descartado. PP-OCRv5 ya funciona. |
| LGPL para el core | Descartado. GPLv3. |
| GPU/CUDA en v1 | Descartado. CPU primero. |
| Métricas a 640×480 / 5 ms | Adaptadas a ROI reales de 1–5 MP y presupuesto por frame. |
| Nombrar tools nuevas | Se añaden `[SHAP]`, `[CAL]`, etc.; no se renombran las existentes. |

---

## 4. Arquitectura objetivo

```
┌──────────────────────────────────────────────────────────┐
│ UI PyQt6 (editor), HMI web, CLI headless                  │
└───────────────────────────┬──────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────┐
│ Python: workflow, tools, resultados, validación           │
│  core/   sin Qt                                           │
│  tools/  lógica + config                                  │
│  accuracy/ verdad sintética y métricas                    │
└───────────────────────────┬──────────────────────────────┘
                            │ pybind11 (solo funciones)
┌───────────────────────────▼──────────────────────────────┐
│ cpp/kernel: pointcloud, plane_fit, [shape_match]          │
└──────────────────────────────────────────────────────────┘
```

### 4.1 Workflow

- `Workflow.tools` es una **lista ordenada**. `Workflow.run()` la recorre en ese orden y, antes de cada tool, copia los outputs de las conexiones a sus inputs.
- No se reordena, no se valida el orden y no se bloquea por ciclos. Un ciclo o una fuente posterior produce el comportamiento que el autor haya definido (output del frame anterior o `None`).
- Se mantiene `stop_on_error` y el estado `last_error`.
- `WorkflowController` es Qt-free y ya permite ejecución síncrona; es la base de `falcocv run`.

### 4.2 Contrato de resultados

Añadir un resultado tipado para medición/inspección sin romper lo existente:

```python
@dataclass
class InspectionResult:
    value: float | None
    unit: str = ""
    tolerance: float | None = None
    verdict: str = ""          # "OK" | "NG" | ""
    score: float | None = None
    uncertainty: float | None = None
    message: str = ""
```

- `ToolResults` sigue siendo la colección genérica.
- Los tools nuevos rellenan el contrato; los antiguos no se tocan.
- El `.cvj` no cambia de esquema.

### 4.3 Kernel

- Funciones puras, sin estado, sin OpenCV.
- Candidatas: `shape_match_vote` (implementada), point cloud y plane fit ya existentes.
- Cada función nueva: benchmark en `benchmarks/`, informe de decisión, test de validación de entrada.

### 4.4 Validación (de calidad, no de pipeline)

- `tests/accuracy/` con generador de escenas y verdad conocida.
- `benchmarks/` para tiempos.
- No se valida el diseño del pipeline; se mide la precisión de los algoritmos.

### 4.5 Headless

```
falcocv run job.cvj --output results.json
falcocv run job.cvj --trace        # opcional: tool, tiempo, error, resumen I/O
```

- Usa `Workflow.run()` con el orden guardado.
- No reordena, no valida, no abre Qt.
- `--trace` es observabilidad, no juicio. Si no se quiere, se elimina.

---

## 5. Hoja de ruta

Estimaciones a **tiempo completo**, con buffer del 20–30 %. Si es a ratos, multiplicar por 1,5–2.

| Fase | Semanas | Entregable | Criterio de salida |
|---|---:|---|---|
| **F0.5** | 1 | `falcocv run` + contrato de resultados + tests | Job ejecutado en CLI, mismo orden, JSON válido |
| **F1** | 2 | Verdad sintética + bake-off del matcher | Informe de decisión |
| **F2** | 6 | ShapeMatch v1/v2 | ≤0,1 px, ≤0,05°, 30 % oclusión, multi-instancia |
| **M1** | 2 | Caliper 2.0, LineFit/CircleFit, fixture | Medición con unidad y repetibilidad |
| **I1** | 2 | Presencia/ausencia, conteo, orientación | Veredicto OK/NG con ShapeMatch |
| **M2** | 3 | Calibración + mm + incertidumbre | `CameraModel` serializado y medición en mm |
| **I2** | 3 | Classifier/anomaly/OCR/códigos | Umbrales, validación y confianza |
| **P1** | 4 | Historial, trazabilidad, export, auditoría | Resultados exportables y consultables |
| **P2** | 5 | PLC/OPC UA/I/O/trigger + empaquetado | Integración industrial y despliegue |

Total: **~6–7 meses** hasta un núcleo con metrología e inspección, más producto.

### Orden de ejecución

1. F0.5.
2. F1.
3. F2 (compartido).
4. Alternar M1 / I1 / M2 / I2.
5. P1 y P2 en paralelo al final.

La alternancia evita dejar una rama parada meses y permite validar con clientes/escenas reales antes de seguir.

---

## 6. F0.5 — Headless y contrato de resultados (1 semana)

### Entregables

- `falcocv run <job.cvj> --output <results.json>`.
- Salida JSON con:

```json
{
  "job": "path/to/job.cvj",
  "ok": true,
  "total_time_ms": 12.3,
  "tools": [
    {"name": "Image Loader", "time_ms": 1.2, "error": null}
  ],
  "results": []
}
```

- Contrato `InspectionResult`/campos de veredicto.
- `--trace` opcional.
- Tests:
  - el orden de la lista se respeta (incluida una fuente posterior);
  - el JSON es válido y contiene tiempos/errores;
  - un job con error devuelve código ≠ 0 y `ok: false`.

### No incluye

- `Workflow.validate()`.
- Detección de ciclos/orden.
- Auto-order.
- Cambios en `.cvj`.

---

## 7. F1 — Verdad sintética y bake-off (2 semanas)

### Generador

Escenas con verdad conocida:

- traslación subpíxel uniforme en [-2, 2] px;
- rotación [-180°, 180°];
- escala [0,8, 1,25];
- ruido gaussiano σ ∈ [0, 20];
- desenfoque σ ∈ [0, 2] px;
- gamma [0,6, 1,6];
- contraste 0,3×–2×;
- oclusión por rectángulos 0–50 %.

### Comparativa

1. `PatternMatcher` actual.
2. `cv2.GeneralizedHoughGuil`.
3. ORB/SIFT + RANSAC.
4. NCC con máscara.

### Métricas

- error de traslación/rotación/escala;
- repetibilidad σ en 30 corridas;
- score vs oclusión;
- tiempo en ROI de 1–5 MP.

### Salida

- `tests/accuracy/` y `benchmarks/bench_matcher.py`.
- informe de decisión con la decisión:
  - si el matcher actual cumple, no se escribe ShapeMatch;
  - si no, se justifica el diseño y se pasa a F2.

### Puerta

No se invierten 6 semanas en ShapeMatch sin este informe.

---

## 8. F2 — ShapeMatch (6 semanas)

### Alcance

- Rotación + traslación, multi-instancia.
- Escala en la segunda iteración; pirámide a media resolución solo como opción (desactivada por precisión).
- Score de cobertura y clutter.
- Oclusión tolerada.
- `act_as_fixture` para encadenar con medición.

### Diseño

- Modelo: features `(x, y, θ, m)` con Sobel + NMS + top-N.
- Búsqueda: bins de orientación + acumulación dispersa.
- Refinamiento: NCC local + ECC afín; fallback de gradiente con distancia + ICP robusto para oclusión.
- Subpíxel: ECC afín + parábola sobre el score de alineación; repetibilidad ≤0,02 px.
- Serialización: arrays comprimidos dentro del `.cvj`, con versión.
- UI: nueva tool `[SHAP]`; `[PATT]` se mantiene.
- Kernel: `shape_match_vote` obligatorio para ShapeMatch; sin fallback Python. Dos solvers explicitos: ECC y oclusion (seleccion por alignment).

### Criterios

| Métrica | Objetivo |
|---|---|
| Traslación | ≤0,1 px |
| Rotación | ≤0,05° |
| Escala | ≤0,1 % |
| Repetibilidad | σ ≤0,02 px |
| Oclusión | score ≥0,7 con 30 % |
| Tiempo | ≤20 ms en 480×360 (comparable a PatternMatcher); ≤50 ms en ROI 1 MP |

---

## 9. Metrología

### M1 — Caliper y geometría

- `EdgeDetection` 2.0: ya tiene perfil 1D, derivada gaussiana y subpíxel; añadir salida de puntos/bordes.
- `LineFit` (TLS + RANSAC) y `CircleFit` (algebraico + RANSAC).
- Fixture encadenado: ShapeMatch/Barcode como fixture; medición en coordenadas de fixture.
- Measurement 2.0: modos distancia/ángulo/área/perímetro/radio/diámetro/anchura/altura con unidad y veredicto.
- Incertidumbre: σ de repetibilidad y tolerancia.

### M2 — Calibración

- `cv2.calibrateCamera` para intrínsecos/distorsión.
- `cv2.calibrateHandEye` para robot/cámara.
- `CameraModel` serializado en el job.
- Conversión px↔mm en Measurement.

---

## 10. Inspección

### I1 — Presencia/ausencia y multi-instancia

- ShapeMatch con `max_matches`.
- Conteo, orientación y posición.
- Veredicto OK/NG y motivo.
- Overlays de acierto/fallo.

### I2 — Clasificación, anomalías, OCR y códigos

- Flujo de entrenamiento/validación de classifier y anomaly.
- Umbrales y confianza.
- OCR/códigos con métricas de calidad.
- Más adelante: inspección por diferencia (golden image), textura.

---

## 11. Producto y despliegue

### P1 — Trazabilidad

- Historial de resultados (SQLite/JSONL).
- Export CSV/JSON.
- Roles, auditoría y marcas de tiempo.
- Integración con HMI.

### P2 — Industrial

- Modbus TCP / OPC UA / I/O digital.
- Trigger de cámara y handshake.
- Empaquetado:
  - Windows: wheel + venv; opción futura de Python embebido/portable.
  - Linux: wheel manylinux + contenedor para servidor/HMI.
  - Sin runtime C++.

---

## 12. Testing y CI

### Capas

- Unitarias: lógica pura.
- Integración: pipelines, jobs, barcode, depth, errores.
- UI: configs, main window, diálogos, ayuda.
- Web: FastAPI, WebSocket, subprocess.
- Empaquetado: wheel, self-test, smoke en venv limpio.
- Accuracy/benchmark: verdad sintética y tiempos (no en cada PR).

### Marcadores

- `ui`, `integration`, `web`, `models`, `slow`, `accuracy`.

### CI

- `full`: Windows + Linux, Python 3.13, suite completa, cobertura y JUnit.
- `compat`: Windows + Linux, Python 3.10, unit + integración.
- `package`: wheel + `check_wheel` + `wheel_smoke`.
- Cobertura: branch, sin `tools/*/config.py`, suelo 35 %. Subir cuando HMI/ML/3D tengan tests.

### Regla para tools nuevas

Toda tool nueva debe traer:

- contrato de serialización;
- test de integración con datos sintéticos;
- test de UI de su config;
- entrada en `tool_help`;
- si toca kernel, benchmark.

---

## 13. Riesgos y decisiones abiertas

| Riesgo | Mitigación |
|---|---|
| ShapeMatch no alcanza precisión | F1 es una puerta; no se invierte antes. |
| Calibración hand-eye sin hardware | Tests sintéticos + checklist manual. |
| Scope creep en inspección | Sprints cerrados; DifferenceInspect fuera de v1. |
| Rendimiento en 5–20 MP | Medir en ROI reales; kernel solo si >1,5×. |
| Python embebido | Se decide después de v1; no bloquea la arquitectura. |
| Cobertura baja en HMI/ML | Tests específicos y mocks; no inflar con tests vacíos. |
| Jobs antiguos | Mantener compatibilidad `.cvj`; no renombrar tools. |

### Decisiones abiertas

- ¿Python embebido en v1 o v1.1?
- ¿DifferenceInspect entra en v1 o después?
- ¿Calibración hand-eye en v1 o v2?
- ¿PLC/OPC UA en v1 o fase posterior?

---

## 14. Próximos pasos

1. **F0.5 completado**: `falcocv run` + contrato de resultados + tests.
2. **F1 completado**: verdad sintética + bake-off; decisión registrada en el informe de la fase.
3. **F2 v1/v2 completado**: rotación/escala/traslación, oclusión y multi-instancia; precisión y tiempo medidos en los informes de ShapeMatch v2 y v3.
4. **F2 v3 completado**: 0,067 px / 0,035 grados / 0,073 %; 15,6 ms en 480x360 y 19,6 ms en 1 MP. Job real 5 MP (template 226x272, ROI 623x671): 90/90, 33,8 ms con +-30/0.9-1.1/All y 18,8 ms con el mismo espacio que PatternMatcher (12,5 ms). El template IC (169x155) queda en 158,4 ms frente a 523 ms de PatternMatcher.
5. Alternar **M1** (caliper/geometría) e **I1** (presencia/ausencia) mientras se decide el kernel.

---

## 15. Tabla de estado

| Fase | Estado | Fecha | Notas |
|---|---|---|---|
| F0.5 Headless + resultados | [x] | 2026-10-27 | `falcocv run` + JSON + tests |
| F1 Verdad sintética + bake-off | [x] | 2026-10-27 | Informe de decisión de la fase |
| F2 ShapeMatch | [x] | 2026-10-27 | v3 con kernel obligatorio, solvers ECC/oclusion explicitos y UI de entrenamiento tipo PatternMatcher |
| M1 Caliper/geometría | [ ] | | |
| I1 Presencia/ausencia | [ ] | | |
| M2 Calibración | [ ] | | |
| I2 Clasificación/OCR | [ ] | | |
| P1 Trazabilidad | [ ] | | |
| P2 Industrial/despliegue | [ ] | | |

---

## 16. Planes eliminados y documentos vivos

- `plans/tools_cpp_porting.md`: **eliminado**. Describía la arquitectura `_cv_core` y el port de tools a C++, descartada. Disponible en el historial de git.
- `plans/falcocv-plan-maestro.md`: **eliminado**. Superado por este documento; disponible en el historial de git.
- `plans/opencv5_opportunities.md`: **no** obsoleto; es un documento vivo de oportunidades de OpenCV 5.
- `audits/` y `reports/`: **internos**, no forman parte del repositorio
  público. Los benchmarks viven en `benchmarks/`.
- `docs/RELEASING.md` y `docs/SMOKE_TEST.md`: documentación operativa. Conservar.
