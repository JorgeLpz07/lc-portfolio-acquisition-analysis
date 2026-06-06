# Métricas Fase 2 — PD (vista consolidada)

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
