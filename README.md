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

### 2. Pipeline (champion actual — con poda de redundantes)

*   **Población LGD:** 27.972 morosos · split 70/30 (`random_state=42`) · train/test con media LGD ~40% / mediana ~29%.
*   **Excluidas (leakage/target):** `saldo_maximo_adeudado`, `LGD_Real`, `target_moroso`.
*   **Redundantes LGD** (multicolinealidad; solo afecta al modelo LGD, no a PD):

| Quitada | Se mantiene |
|---------|-------------|
| `tasa_interes` | `grado` |
| `flag_meses_desde_ultima_mora_bancaria_na` | `flag_meses_desde_ultima_mora_na` |
| `flag_meses_desde_ultima_mora_revolvente_na` | ↑ |
| `num_lineas_credito_abiertas` | `num_total_lineas_credito` |
| `meses_desde_ultima_mora_bancaria` | `meses_desde_ultima_mora` |
| `meses_desde_ultima_mora_revolvente` | ↑ |
| `codigo_region` (categórica) | `estado_residencia` |

*   **Predictores finales:** 21 numéricas + 6 categóricas (`cat_cols_lgd`) → **90** columnas tras One-Hot + `StandardScaler` en numéricas (`scaler_lgd`).
*   **Predicción:** acotada a **[0, 100]%** en inferencia.

### 3. Métricas (MAE y RMSE — criterio del enunciado)

Mismo split 70/30 (`random_state=42`) · 27.972 morosos · Regresión Lineal con `clip(0, 100)` en predicción.

#### Champion actual (con poda)

| Modelo | MAE | RMSE |
|--------|-----|------|
| Baseline (media train) | 24,20 pp | 43,37 pp |
| **Regresión Lineal** | **17,90 pp** | **34,96 pp** |
| Mejora vs baseline | −6,30 pp | −8,41 pp |

#### Comparativa: sin poda vs con poda

| | Sin poda | Con poda (champion) | Δ (poda − sin poda) |
|--|----------|---------------------|---------------------|
| **Predictores** | 27 num + 7 cat | 21 num + 6 cat | −6 num, −1 cat |
| **Columnas tras One-Hot** | 106 | 90 | −16 |
| **Baseline MAE** | 24,20 pp | 24,20 pp | 0,00 pp |
| **Regresión Lineal MAE** | 17,91 pp | **17,90 pp** | −0,01 pp |
| **Baseline RMSE** | 43,37 pp | 43,37 pp | 0,00 pp |
| **Regresión Lineal RMSE** | 34,98 pp | **34,96 pp** | −0,02 pp |
| **Mejora MAE vs baseline** | −6,29 pp | −6,30 pp | −0,01 pp |
| **Mejora RMSE vs baseline** | −8,39 pp | −8,41 pp | −0,02 pp |

**Conclusión:** la poda **no cambia el rendimiento** de forma material (Δ < 0,02 pp en MAE y RMSE), pero **reduce 16 columnas** y hace más legibles los coeficientes (menos geografía redundante).

**Top drivers (tras poda):** `proposito_prestamo` (wedding, medical, educational ↑ severidad), **EAD** ↓ severidad (préstamos grandes pierden menor %), estados con pocos morosos (NH, MS, ME…). Ver gráfica top 20 en `model.ipynb`.

**Validación:** en morosos típicos el error es bajo; los peores casos son préstamos pequeños (EAD ~$1.000) con LGD real >600% — inflan RMSE, impacto marginal en EL. Corrida al 100% pendiente.

Detalle ampliado LGD: **[metricas/README.md](metricas/README.md)** (sección Fase 3).

---

## 📚 Lecciones aprendidas

### Datos y muestreo
*   **`PORCENTAJE_DATOS` unificado:** Notebook al **10%** (`0.1`) en Fases 2–4; una corrida documentada al 20% fue error de configuración y se descartó. Corrida al 100% prevista después.
*   **`stratify` en LGD:** No aplica sobre `target_moroso` (todos son morosos). No sustituye quitar variables redundantes. Comprobamos representatividad con **media/mediana de `LGD_Real` en train vs test** (40,9% / 29,3% vs 40,4% / 28,7%) — suficiente sin `stratify`.

### PD vs LGD — modelos distintos
*   **PD:** clasificación → Regresión **Logística** → probabilidad de impago.
*   **LGD:** regresión continua (%) → Regresión **Lineal** → severidad. Misma familia “lineal” por **explicabilidad**, no porque ambos sean el mismo problema.

### Target LGD vs predictores
*   `LGD_Real` se calcula con `saldo_maximo_adeudado`, pero esa columna **no entra** en el modelo (leakage). El histórico define el target; la predicción usa solo variables de originación.

### Outliers y métricas LGD
*   Target con máximos >100% (hasta ~780% en EAD bajos) inflan **RMSE** más que **MAE**. Predicciones acotadas con **`clip(0, 100)`** en inferencia.
*   MAE **17,90 pp** vs baseline **24,20 pp** (−26%) valida el modelo.

### Poda de redundantes LGD
*   Quitamos 6 numéricas redundantes + `codigo_region`; MAE/RMSE casi iguales (Δ < 0,02 pp).
*   **`stratify` no reduce ruido** — la multicolinealidad se resuelve **excluyendo columnas**, no estratificando el split.

### Coeficientes LGD — gráfica top 20 (tras poda)
*   **Antes de podar:** top dominado por geografía + `codigo_region` (coef. ~60–82 pp), poco interpretable.
*   **Tras podar:** destacan **`proposito_prestamo`** (wedding ~+30 pp, medical/educational ~+22 pp), **EAD** (~−24 pp) y algunos estados con pocos morosos.
*   Estados extremos (NH, MS, ME…) requieren cautela: bajo volumen en train → coeficientes inestables.
*   **Lección:** validar LGD con **MAE/RMSE**; narrativa de negocio en **PD**; en LGD priorizar drivers económicos (EAD, propósito) sobre geografía fina.

### Validación caso a caso
*   Aciertos casi perfectos en LGD típico (8%–62%). Peores casos: EAD ~$1.000 y LGD real >600% — pocos préstamos, impacto marginal en EL total.

### EL y calibración (Fase 4)
*   El motor ML **ordena bien** el riesgo (morosos con PD alta → EL alta), pero la **PD media sin calibrar** (~44%) supera la morosidad real en test (~13%) por `class_weight='balanced'`.
*   **No presentar el EL del motor en bruto** como cifra de compra; anclar la decisión en **históricos actuariales** y usar el ML para **segmentar** y explicar.
*   Corrida al **100%** pendiente para replicar estas cifras en cartera completa.

---

## 🎯 Fase 4: Pérdida Esperada (EL)

> **Alcance:** notebook con `PORCENTAJE_DATOS = 0.1` · test PD = 64.190 préstamos (30%) · **corrida al 100% pendiente**.

Fórmula: `EL = PD × (LGD / 100) × EAD` · PD = Regresión Logística · LGD = Regresión Lineal (poda, `clip` 0–100%).

### Motor ML — resultado técnico (test PD)

| Concepto | Valor |
|----------|-------|
| Préstamos test | 64.190 |
| Morosos reales en test | 8.392 (**13,1%**) |
| PD media predicha | **0,443** |
| LGD media predicha | **45,0%** (mediana 47,2%) |
| EAD total test | **946,94 M$** |
| **EL total motor** | **130,24 M$** |
| **EL / EAD** | **13,75%** |

**Frase técnica del notebook:** *"De una cartera test valorada en **946,94 millones** de dólares, el motor estima **130,24 millones** de pérdida esperada."*

### Validación: ¿qué tan creíble es frente a lo observado?

Tres enfoques sobre el **mismo test** (mismo split `random_state=42`):

| Enfoque | Método | EL total | % sobre EAD | Uso recomendado |
|---------|--------|----------|-------------|-----------------|
| **Pérdida realizada** | Solo morosos: `LGD_Real × EAD` (lo que ya pasó) | **~39 M$** | **~4,1%** | Validación retrospectiva |
| **Actuarial estándar** | Morosidad test (13,1%) × LGD medio morosos (~42%) × EAD | **~52 M$** | **~5,5%** | Ancla para comité / dólares |
| **Motor ML (sin calibrar)** | `predict_proba` × LGD predicho × EAD | **130 M$** | **13,75%** | Ranking y granularidad |

El motor ML **sobreestima ~3,4×** la pérdida realizada en test. La causa principal es la **PD no calibrada** (media 44% vs morosidad real 13%), no un error en la fórmula EL.

### Postura ante el comité

| Pregunta | Respuesta |
|----------|-----------|
| ¿El motor está bien construido? | Sí — fórmula estándar, modelos explicables, sin leakage. |
| ¿130 M$ es lo que perdimos? | No — lo realizado fue **~39 M$**. |
| ¿Qué cifra usar para decidir? | **Actuarial / histórico (~52 M$)** como magnitud; ML para **priorizar** préstamos. |
| ¿Para qué sirve el DS? | Segmentación, explicabilidad y velocidad — no sustituye el análisis en dólares calibrado. |

**Próximos pasos antes de corrida completa:** recalibrar PD (Platt / isotónica) o escalar EL motor por factor `morosidad_real / PD_media`; replicar tabla anterior al **100%** (`PORCENTAJE_DATOS = 1.0`).
