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

---
*Siguiente Paso: Fase 2 - Modelado de Probabilidad de Default (PD)*
