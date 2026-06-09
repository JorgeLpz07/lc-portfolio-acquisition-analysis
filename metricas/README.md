# Métricas del motor de riesgo — PD y LGD

Detalle ampliado, históricos JSON y comparativas entre corridas.  
**Resumen ejecutivo:** [`../README.md`](../README.md) · **Lecciones aprendidas:** [`../README.md`](../README.md) (sección 📚)

---

## Fase 2 — PD (vista consolidada)

Dataset: **2.139.643 filas** · Test: **641.893 préstamos** · Umbral decisión: **0.50**

---

## Resumen de corridas

| ID | Fecha (UTC) | Modelos | Cols. redundantes | Features | Duración |
|----|-------------|---------|-------------------|----------|----------|
| **01** | 2026-06-06 05:41 | 4 (RandomizedSearchCV) | Eliminadas | 107 | ~35 min |
| **02** | 2026-06-06 05:48 | Solo Regresión Logística | Conservadas | 112 | ~3 min |

Datos fuente (JSON): `historico_01_4modelos_sin_redundantes.json` · `historico_02_regresion_con_redundantes.json`

---

## Histórico 01 — Comparativa completa (4 modelos, sin redundantes)

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC | Gini | Gini > 0.45 |
|--------|----------|-----------|--------|-----|---------|------|-------------|
| XGBoost | 65.5% | 23.1% | **70.3%** | 0.347 | **0.740** | **0.479** | Sí |
| Árbol de Decisión | 60.7% | 21.4% | **74.9%** | 0.332 | 0.731 | 0.462 | Sí |
| **Regresión Logística** | **66.9%** | 23.1% | 66.0% | 0.343 | 0.727 | 0.454 | Casi |
| Random Forest | 64.5% | 22.1% | 68.1% | 0.334 | 0.720 | 0.441 | No |

Modelos con Gini > 0.45: **3**

### Cómo leer estas métricas

- **Accuracy** mide aciertos totales; con ~87% de buenos pagadores puede ser alta aunque se pasen morosos.
- **Precision** (~23%): de cada 100 marcados como morosos, ~77 son buenos clientes (falsas alarmas).
- **Recall**: % de morosos reales que detectamos. Prioridad de negocio en este proyecto.
- **ROC-AUC / Gini**: capacidad de **ordenar** riesgo (estándar bancario: Gini > 0.45).
- **F1**: equilibrio precision–recall; bajo aquí refleja el desbalanceo de clases.

---

## Histórico 02 — Regresión Logística para presentación (con redundantes)

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC | Gini | Gini > 0.45 |
|--------|----------|-----------|--------|-----|---------|------|-------------|
| **Regresión Logística** | **67.0%** | **23.2%** | 65.9% | **0.343** | **0.728** | **0.456** | **Sí** |

---

## Regresión Logística — Comparación directa (01 vs 02)

| Métrica | Hist. 01 (sin redundantes) | Hist. 02 (con redundantes) | Δ |
|---------|---------------------------|---------------------------|-----|
| Accuracy | 66.90% | **67.04%** | +0.14 pp |
| Precision | 23.14% | **23.21%** | +0.07 pp |
| Recall | **65.99%** | 65.90% | −0.09 pp |
| F1 | 0.343 | **0.343** | ≈ |
| ROC-AUC | 0.727 | **0.728** | +0.001 |
| Gini | 0.454 | **0.456** | +0.002 |

Con redundantes conservadas, la LR **mejora ligeramente** en Accuracy, Precision y Gini, con un recall prácticamente igual.

---

## Matriz de confusión — Regresión Logística

| | Hist. 01 | Hist. 02 |
|--|----------|----------|
| Buenos detectados (TN) | 374.056 | **375.024** |
| Morosos evitados (TP) | **55.375** | 55.298 |
| Morosos escapados (FN) | **28.540** | 28.617 |
| Buenos rechazados (FP) | 183.922 | **182.954** |

---

## Lectura para el comité

| Pregunta | Respuesta con estos datos |
|----------|---------------------------|
| ¿Qué modelo discrimina mejor? | **XGBoost** (Gini 0.479, hist. 01) |
| ¿Cuál presentamos primero? | **Regresión Logística** (hist. 02): Gini 0.456, coeficientes explicables |
| ¿Accuracy engaña? | Sí, parcialmente: RF y XGBoost tienen Accuracy menor que LR pero mejor Recall/Gini |
| ¿Eliminar redundantes? | LR gana marginalmente **manteniéndolas**; la poda fue por multicolinealidad, no por caída fuerte de métricas |

---

## Champion recomendado por objetivo

| Objetivo | Corrida | Modelo |
|----------|---------|--------|
| Máxima discriminación (AUC/Gini) | 01 | XGBoost |
| Explicabilidad regulatoria | 02 | Regresión Logística |
| Máximo recall (detectar morosos) | 01 | Árbol de Decisión (74.9%, peor Accuracy) |

---

## Fase 3 — LGD (Severidad)

**Notebook:** `model.ipynb` · **Champion:** Regresión Lineal (alineado con PD para explicabilidad)

**Configuración notebook (unificada con Fases 2–4):** `PORCENTAJE_DATOS = 0.1` (~214k filas) · `test_size = 0.3` · `random_state = 42`

### Población y target

| Concepto | Valor (esperado con 0.1) |
|----------|---------------------------|
| Cartera total | ~214.000 préstamos |
| Morosos (población LGD) | ~28.000 (~13%) |
| Target | `LGD_Real = (saldo_maximo_adeudado / EAD) × 100` |

**Distribución típica de `LGD_Real`:** mediana < media (cola derecha); máximos históricos >100% posibles por intereses/moras acumuladas.

### Pipeline de modelado (champion actual — con poda)

| Paso | Decisión |
|------|----------|
| Morosos | 27.972 · split 70/30 · `random_state=42` |
| Predictores | **21 numéricas + 6 categóricas** (`cat_cols_lgd`) → **90** tras One-Hot |
| Excluidas (leakage/target) | `saldo_maximo_adeudado`, `LGD_Real`, `target_moroso` |
| Redundantes LGD (solo Fase 3) | `tasa_interes`, 2 flags de mora NA, `num_lineas_credito_abiertas`, 2 meses mora bancaria/revolvente, `codigo_region` |
| Escalado | `StandardScaler` solo en numéricas (`scaler_lgd`, independiente de PD) |
| Post-proceso | Predicciones acotadas a **[0, 100]%** (criterio económico en inferencia) |
| Target histórico | **Sin recorte** — fiel a la fórmula del enunciado |

### Métricas en test (criterio del enunciado: MAE y RMSE)

| Modelo | MAE (pp) | RMSE (pp) |
|--------|----------|-----------|
| Baseline (predecir media train) | 24,20 | 43,37 |
| **Regresión Lineal (champion)** | **17,90** | **34,96** |
| Mejora vs baseline | −6,30 | −8,41 |

**Comparativa poda** (mismo split):

| Versión | Pre-dummies | MAE | RMSE |
|---------|-------------|-----|------|
| Sin poda | 27 num + 7 cat → 106 | 17,91 | 34,98 |
| **Con poda (actual)** | **21 num + 6 cat → 90** | **17,90** | **34,96** |

Corrida al 100% pendiente.

- **RMSE > MAE** es patrón esperado por outliers en el target.
- Champion: **Regresión Lineal** (explicabilidad; RF descartado aunque rinda mejor en MAE).
- **Top coeficientes (tras poda):** `proposito_prestamo` (wedding, medical, educational), **EAD** (negativo), estados con bajo volumen.

### Validación caso a caso (test)

**Predicciones más acertadas** — LGD real entre ~8% y ~62%, error ≈ 0 pp (ej. EAD 7k–22,5k, distintos grados).

**Peores predicciones** — patrón común:

| EAD | LGD real | LGD predicho | Error |
|-----|----------|--------------|-------|
| $1.000 | 655% – 780% | 78% – 100% | hasta ~689 pp |

Causa: préstamos pequeños con saldo máximo muy superior al capital originado. El modelo no extrapola a 780%; el `clip` limita la salida a 100%. Son **pocos casos** pero inflan el RMSE.

**Decisión documentada:** no recortar el target en Fase 1; el impacto en EL total de cartera se espera marginal (EAD bajo y baja frecuencia). Alternativa futura: cap `LGD_Real` a 100% para alinear con estándar bancario y bajar RMSE.

### Lectura para el comité — Fase 3

| Pregunta | Respuesta |
|----------|-----------|
| ¿El modelo aporta vs adivinar la media? | Sí — MAE baja 26% (24,2 → 17,9 pp) |
| ¿Hay leakage? | No — `saldo_maximo_adeudado` excluido; MAE no sospechosamente bajo |
| ¿Por qué RMSE alto? | Outliers históricos con LGD > 100% |
| ¿Poda redundantes? | Sí — sin pérdida material de MAE/RMSE; coeficientes más legibles |
| ¿Champion? | **Regresión Lineal** — MAE/RMSE mejores que baseline |
| ¿Listo para Fase 4? | Sí — `modelo_lgd_champion` + `scaler_lgd` + pipeline 90 columnas |

### Variables clave guardadas para Fase 4

`modelo_lgd_champion` · `scaler_lgd` · `num_cols_lgd` · `cat_cols_lgd` · columnas de `X_train_lgd`
