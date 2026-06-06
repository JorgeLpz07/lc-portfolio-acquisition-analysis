# 📈 Lending Club: Portfolio Acquisition Analysis & Risk Modeling

Este proyecto realiza un análisis exhaustivo y modelado de riesgo para la adquisición de una cartera de préstamos de **Lending Club**. El objetivo es determinar la rentabilidad y el riesgo de impago para optimizar la estrategia de inversión.

## 🛠️ Fase 1: Preparación de Datos y Business Intelligence

En esta fase inicial, hemos transformado los datos crudos en un dataset robusto para modelado financiero:

### 1. Gestión de Datos y Memoria
*   **Muestreo Estratificado:** Implementamos un muestreo del 10% de la cartera original garantizando la representatividad de la clase minoritaria (`target_moroso`).
*   **Limpieza de Data Leakage:** Eliminamos variables post-originación (como pagos recibidos o intereses acumulados) para evitar que el modelo "vea el futuro" y genere resultados artificialmente perfectos.

### 2. Ingeniería de Variables de Riesgo
*   **Historial Crediticio:** Calculamos la antigüedad de las líneas de crédito en años para medir la madurez financiera del prestatario.
*   **EAD (Exposure at Default):** Definida como el monto total financiado al momento de la emisión.
*   **LGD Real (Loss Given Default):** Calculada como la severidad de la pérdida histórica basada en el saldo máximo adeudado frente al monto original.

### 3. Tratamiento de Outliers (Capping)
*   Aplicamos **Winsorización (Capping)** mediante la regla de 1.5x IQR para variables de escala (Ingreso Anual, Saldos Revolventes). 
*   **Justificación:** Esto protege al modelo de distorsiones por valores extremos sin perder la información de los registros, manteniendo la estabilidad de las distribuciones.

## 📊 Fase 2: Clasificación de Riesgo (PD)

### Baseline histórico (ejecución anterior — dataset completo)

Configuración: `PORCENTAJE_DATOS = 1.0`, `MODO_COMPLETO = False` (solo Regresión Logística).

| Métrica | Regresión Logística |
|---------|---------------------|
| ROC-AUC | 0.728 |
| Gini | 0.455 |
| Recall | 66% |

Champion provisional: **Regresión Logística** (Gini > 0.45, criterio regulatorio).

Impacto en negocio (test, ~641.893 clientes): 55.267 impagos evitados · 28.648 falsos negativos · 374.999 aprobados correctamente.

---

### Registro de métricas — 2025-06-05 (tras limpieza de features)

Configuración: `PORCENTAJE_DATOS = 0.1` (~214k filas), `MODO_COMPLETO = True` (4 modelos + RandomizedSearchCV).

Cambios respecto al baseline: flags en variables de tiempo, eliminación de redundantes por multicolinealidad, sin variables de leakage.

**Comparativa final de modelos**

| Modelo | ROC-AUC | Gini | Recall |
|--------|---------|------|--------|
| XGBoost | 0.734880 | 0.469761 | 0.690062 |
| Regresión Logística | 0.724619 | 0.449238 | 0.658365 |
| Random Forest | 0.717236 | 0.434471 | 0.632626 |
| Árbol de Decisión | 0.704625 | 0.409251 | 0.743446 |

Modelos que superan el estándar (> 0.45 Gini): **1** (XGBoost).

**Métricas detalladas (sin re-entrenar)**

| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|----------|-----------|--------|----------|---------|
| XGBoost | 0.660991 | 0.232095 | 0.690062 | 0.347359 | 0.734880 |
| Regresión Logística | 0.668889 | 0.231056 | 0.658365 | 0.342063 | 0.724619 |
| Random Forest | 0.677053 | 0.231269 | 0.632626 | 0.338714 | 0.717236 |
| Árbol de Decisión | 0.575386 | 0.199062 | 0.743446 | 0.314038 | 0.704625 |

> **Nota:** Estas cifras pueden variar al ejecutar con `PORCENTAJE_DATOS = 1.0` u otros ajustes de pipeline. Conservar este registro como referencia antes de la corrida final.

---
## 🛠️ Próximos Pasos: Fase 3 - LGD
El siguiente objetivo es predecir la **Severidad de la Pérdida (Loss Given Default)**. Utilizaremos modelos de regresión para estimar qué porcentaje del capital se recupera realmente en los 28,648 casos donde el modelo de PD falló, permitiéndonos calcular la **Pérdida Esperada (EL)** final.