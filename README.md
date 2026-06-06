# 📈 Lending Club: Portfolio Acquisition Analysis & Risk Modeling

Este proyecto realiza un análisis exhaustivo y modelado de riesgo para la adquisición de una cartera de préstamos de **Lending Club**. El objetivo es determinar la rentabilidad y el riesgo de impago para optimizar la estrategia de inversión.

El objetivo final del motor es estimar la **Pérdida Esperada (EL)**:

$$\text{EL} = \text{PD} \times \text{LGD} \times \text{EAD}$$

**Configuración del notebook (`model.ipynb`):** `PORCENTAJE_DATOS = 0.1` (~214k filas, muestreo estratificado). Fases 2–4 usan el mismo muestreo. Las corridas batch al 100% en `metricas/` son referencia aparte (`run_fase2_metrics.py`).

---

## 🛠️ Fase 1: Preparación de Datos y Business Intelligence

En esta fase inicial, hemos transformado los datos crudos en un dataset robusto para modelado financiero:

### 1. Gestión de Datos y Memoria
*   **Muestreo Estratificado:** Implementamos un muestreo del **10%** de la cartera original (`PORCENTAJE_DATOS = 0.1`) garantizando la representatividad de la clase minoritaria (`target_moroso`).
*   **Limpieza de Data Leakage:** Eliminamos variables post-originación (como pagos recibidos o intereses acumulados) para evitar que el modelo "vea el futuro" y genere resultados artificialmente perfectos.

### 2. Ingeniería de Variables de Riesgo
*   **Historial Crediticio:** Calculamos la antigüedad de las líneas de crédito en años para medir la madurez financiera del prestatario.
*   **EAD (Exposure at Default):** Definida como el monto total financiado al momento de la emisión.
*   **LGD Real (Loss Given Default):** Calculada como la severidad de la pérdida histórica basada en el saldo máximo adeudado frente al monto original: `(saldo_maximo_adeudado / EAD) × 100`.

### 3. Tratamiento de Outliers (Capping)
*   Aplicamos **Winsorización (Capping)** mediante la regla de 1.5x IQR para variables de escala (Ingreso Anual, Saldos Revolventes).
*   **Justificación:** Esto protege al modelo de distorsiones por valores extremos sin perder la información de los registros, manteniendo la estabilidad de las distribuciones y de la métrica `LGD_Real` para la Fase 3.

---

## 📊 Fase 2: Clasificación de Riesgo (PD)

Necesitamos separar a los buenos clientes de los tóxicos. **Champion:** Regresión Logística (explicable ante regulador; priorizamos Recall y Gini sobre Accuracy).

### Resultado — notebook (`PORCENTAJE_DATOS = 0.1`)

Champion: **Regresión Logística** · `MODO_COMPLETO` según panel de control del notebook.

> Métricas numéricas: **actualizar tras re-ejecutar** el notebook con `PORCENTAJE_DATOS = 0.1` (corrida unificada Fases 2–4).

### Referencia batch — dataset completo (`PORCENTAJE_DATOS = 1.0`)

Corrida en `run_fase2_metrics.py` (solo LR, presentación). Resumen:

| Métrica | Regresión Logística |
|---------|---------------------|
| ROC-AUC | 0,728 |
| Gini | 0,456 |
| Recall | 65,9% |

Test: 641.893 préstamos · Gini > 0,45. Detalle e histórico 4 modelos: **[metricas/README.md](metricas/README.md)**

---

## 📉 Fase 3: Severidad de la Pérdida (LGD)

Si el cliente ya impagó, estimamos **qué porcentaje del capital se pierde**. Solo entrenamos con morosos (`target_moroso = 1`). **Champion:** Regresión Lineal (coherente con PD).

### 1. Población y target (notebook, `PORCENTAJE_DATOS = 0.1`)
*   **Cartera:** ~214k préstamos · ~13% morosos (mismo muestreo que Fase 2).
*   **Target `LGD_Real`:** `(saldo_maximo_adeudado / EAD) × 100` · sin nulos.
*   **Outliers en target:** casos con LGD > 100% (saldo máximo >> EAD; intereses/moras). Se documentan; no se recortan en entrenamiento; predicciones acotadas a [0, 100]%.

### 2. Pipeline
*   **Excluidas del modelo:** `saldo_maximo_adeudado` (leakage), `LGD_Real`, `target_moroso`.
*   **Preproceso:** One-Hot (106 columnas) + `StandardScaler` en numéricas (`scaler_lgd`).
*   **Predicción:** acotada a **[0, 100]%** en inferencia.

### 3. Métricas (MAE y RMSE — criterio del enunciado)

Champion: **Regresión Lineal** vs baseline (predecir media del train).

> Valores numéricos: **actualizar tras re-ejecutar** con `PORCENTAJE_DATOS = 0.1`. Una corrida previa al 20% fue descartada (error de configuración).

**Validación (insight estable):** en morosos típicos el error es bajo; los peores casos son préstamos pequeños (EAD ~$1.000) con LGD real >600% — inflan RMSE, impacto marginal en EL.

Detalle ampliado LGD: **[metricas/README.md](metricas/README.md)** (sección Fase 3).

---

## 🎯 Próximo paso: Fase 4 — Pérdida Esperada (EL)

Unir en el **test completo de PD** (todos los préstamos):

1. PD ← Regresión Logística (`predict_proba`)
2. LGD ← Regresión Lineal (`predict`, clip 0–100%)
3. `EL = PD × (LGD / 100) × EAD` por préstamo → sumar total

**Entregable:** *"De una cartera test valorada en [X] millones, estimamos pérdidas por impago de [Y] millones."*
