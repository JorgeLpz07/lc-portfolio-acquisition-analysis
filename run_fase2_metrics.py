"""Ejecuta Fase 2 (PD) y guarda métricas en archivos independientes."""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
PORCENTAJE_DATOS = 1.0
MODO_COMPLETO = False
ELIMINAR_COLS_REDUNDANTES = False
RANDOM_STATE = 42
TEST_SIZE = 0.3
UMBRAL = 0.50
CHAMPION_CM = "Regresión Logística"
METRICAS_DIR = ROOT / "metricas"


def output_slug() -> str:
    modelo = "4modelos" if MODO_COMPLETO else "regresion"
    cols = "sin_redundantes" if ELIMINAR_COLS_REDUNDANTES else "con_redundantes"
    return f"{modelo}_{cols}"


def log(msg: str) -> None:
    print(msg, flush=True)


def load_and_prepare() -> tuple[pd.DataFrame, list[str], list[str]]:
    traducciones = {
        "funded_amnt": "monto_financiado",
        "interest_rate": "tasa_interes",
        "monthly_payment": "pago_mensual",
        "grade": "grado",
        "loan_term_months": "plazo_prestamo_meses",
        "loan_purpose": "proposito_prestamo",
        "disbursement_method": "metodo_desembolso",
        "issue_date_month": "mes_emision",
        "issue_date_year": "anio_emision",
        "emp_title": "titulo_empleo",
        "emp_length": "antiguedad_empleo",
        "home_ownership_status": "estado_vivienda",
        "annual_income": "ingreso_anual",
        "verification_status": "estado_verificacion",
        "addr_state": "estado_residencia",
        "region_code": "codigo_region",
        "dept_paym_income_ratio": "ratio_deuda_ingresos",
        "num_30+_delinq_in_2yrs": "num_moras_30_dias_2anios",
        "num_inq_in_6mths": "num_consultas_6meses",
        "num_inq_in_12mths": "num_consultas_12meses",
        "num_inq": "num_consultas_totales",
        "mths_since_last_delinq": "meses_desde_ultima_mora",
        "num_open_credit_lines": "num_lineas_credito_abiertas",
        "num_derogatory_pub_rec": "num_registros_negativos",
        "total_credit_revolving_bal": "saldo_revolvente_total",
        "used_credit_share": "porcentaje_credito_utilizado",
        "tot_num_credit_lines": "num_total_lineas_credito",
        "initial_list_status": "estado_listado_inicial",
        "earliest_cr_line_month": "mes_primera_linea_credito",
        "earliest_cr_line_year": "anio_primera_linea_credito",
        "remaining_princ_for_tot_amnt_fund": "capital_pendiente_total",
        "paym_rec_for_tot_amnt_fund": "pagos_recibidos_total",
        "princ_rec": "capital_recibido",
        "interest_rec": "intereses_recibidos",
        "late_fees_rec": "comisiones_mora_recibidas",
        "max_bal_owed": "saldo_maximo_adeudado",
        "bal_to_cred_lim": "saldo_vs_limite_credito",
        "num_open_trades_in_6mths": "num_operaciones_abiertas_6meses",
        "num_installment_acc_op_in_12mths": "num_cuentas_cuotas_12meses",
        "num_installment_acc_op_in_24mths": "num_cuentas_cuotas_24meses",
        "mths_since_last_installment_acc_op": "meses_desde_ultima_cuenta_cuotas",
        "num_rev_trades_op_in_12mths": "num_operaciones_revolventes_12meses",
        "num_rev_trades_op_in_24mths": "num_operaciones_revolventes_24meses",
        "mths_since_recent_bankcard_delinq": "meses_desde_ultima_mora_bancaria",
        "mths_since_recent_revol_delinq": "meses_desde_ultima_mora_revolvente",
        "y": "target_moroso",
    }

    log("Cargando datos...")
    y = pd.read_csv(ROOT / "data" / "target.csv")
    df_full = pd.read_csv(ROOT / "data" / "x.csv")
    df_full = pd.concat([df_full, y], axis=1)
    df_full.rename(columns=traducciones, inplace=True)

    if PORCENTAJE_DATOS < 1.0:
        df, _ = train_test_split(
            df_full,
            train_size=PORCENTAJE_DATOS,
            stratify=df_full["target_moroso"],
            random_state=RANDOM_STATE,
        )
        log(f"Muestra {PORCENTAJE_DATOS * 100}%: {len(df):,} filas")
    else:
        df = df_full.copy()
        log(f"Dataset completo: {len(df):,} filas")

    df = df.reset_index(drop=True)

    df["saldo_maximo_adeudado"] = df.groupby("grado")["saldo_maximo_adeudado"].transform(
        lambda x: x.fillna(x.median())
    )
    df["saldo_vs_limite_credito"] = df.groupby("grado")["saldo_vs_limite_credito"].transform(
        lambda x: x.fillna(x.median())
    )

    vars_tiempo = [
        "meses_desde_ultima_mora_bancaria",
        "meses_desde_ultima_mora_revolvente",
        "meses_desde_ultima_mora",
        "meses_desde_ultima_cuenta_cuotas",
    ]
    for col in vars_tiempo:
        df[f"flag_{col}_na"] = df[col].isna().astype(int)
        df[col] = df[col].fillna(0)

    vars_conteo = [
        "num_consultas_totales",
        "num_operaciones_abiertas_6meses",
        "num_operaciones_revolventes_12meses",
        "num_operaciones_revolventes_24meses",
        "num_cuentas_cuotas_12meses",
        "num_cuentas_cuotas_24meses",
    ]
    df[vars_conteo] = df[vars_conteo].fillna(0)

    df["titulo_empleo"] = df["titulo_empleo"].fillna("Unknown")
    df["antiguedad_empleo"] = df["antiguedad_empleo"].fillna("Unknown")

    df["anios_historial_crediticio"] = df["anio_emision"] - df["anio_primera_linea_credito"]
    df.drop(
        columns=["anio_emision", "anio_primera_linea_credito", "mes_emision", "mes_primera_linea_credito"],
        inplace=True,
    )

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if "codigo_region" in num_cols:
        num_cols.remove("codigo_region")
        cat_cols.append("codigo_region")
    num_cols.remove("target_moroso")

    cols_leakage = [
        "capital_pendiente_total",
        "pagos_recibidos_total",
        "capital_recibido",
        "intereses_recibidos",
        "comisiones_mora_recibidas",
    ]
    df.drop(columns=cols_leakage, inplace=True)
    num_cols = [c for c in num_cols if c not in cols_leakage]

    variables_a_cappear = [
        "ingreso_anual",
        "saldo_revolvente_total",
        "saldo_maximo_adeudado",
        "pago_mensual",
        "num_total_lineas_credito",
        "monto_financiado",
    ]
    for col in variables_a_cappear:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lim_inf = max(df[col].min(), q1 - 1.5 * iqr)
        lim_sup = q3 + 1.5 * iqr
        df[col] = df[col].clip(lower=lim_inf, upper=lim_sup)

    df.rename(columns={"monto_financiado": "EAD"}, inplace=True)
    df["LGD_Real"] = (df["saldo_maximo_adeudado"] / df["EAD"]) * 100

    mapa_antiguedad = {
        "10+ years": 10, "9 years": 9, "8 years": 8, "7 years": 7, "6 years": 6,
        "5 years": 5, "4 years": 4, "3 years": 3, "2 years": 2, "1 year": 1,
        "< 1 year": 0, "Unknown": -1,
    }
    df["antiguedad_empleo"] = df["antiguedad_empleo"].map(mapa_antiguedad)
    grados_ordenados = sorted(df["grado"].unique())
    mapa_grado = {grado: i for i, grado in enumerate(grados_ordenados)}
    df["grado"] = df["grado"].map(mapa_grado)

    df.drop(columns=["titulo_empleo"], inplace=True)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if "codigo_region" in num_cols:
        num_cols.remove("codigo_region")
        cat_cols.append("codigo_region")
    num_cols.remove("target_moroso")

    if ELIMINAR_COLS_REDUNDANTES:
        cols_redundantes = [
            "pago_mensual",
            "num_cuentas_cuotas_12meses",
            "num_operaciones_revolventes_12meses",
            "num_consultas_12meses",
            "num_consultas_totales",
        ]
        num_cols = [c for c in num_cols if c not in cols_redundantes]

    return df, num_cols, cat_cols


def train_models(X_train, X_test, y_train, y_test) -> dict:
    ratio_balanceo = (y_train == 0).sum() / (y_train == 1).sum()
    mejores_modelos = {}

    if MODO_COMPLETO:
        config_modelos = {
            "Regresión Logística": {
                "modelo": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
                "params": {"C": [0.1, 1, 10], "solver": ["lbfgs", "liblinear"]},
            },
            "Árbol de Decisión": {
                "modelo": DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
                "params": {"max_depth": [5, 10, 15], "min_samples_split": [2, 10]},
            },
            "Random Forest": {
                "modelo": RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE),
                "params": {"n_estimators": [100, 200], "max_depth": [5, 10]},
            },
            "XGBoost": {
                "modelo": XGBClassifier(
                    scale_pos_weight=ratio_balanceo,
                    random_state=RANDOM_STATE,
                    eval_metric="logloss",
                ),
                "params": {"max_depth": [3, 5], "learning_rate": [0.01, 0.1]},
            },
        }
        for nombre, config in config_modelos.items():
            log(f"Optimizando {nombre}...")
            t0 = time.time()
            search = RandomizedSearchCV(
                config["modelo"],
                config["params"],
                n_iter=10,
                cv=3,
                scoring="roc_auc",
                n_jobs=4,
                verbose=0,
                random_state=RANDOM_STATE,
            )
            search.fit(X_train, y_train)
            mejores_modelos[nombre] = search.best_estimator_
            log(f"  {nombre} listo en {time.time() - t0:.0f}s")
    else:
        modelo_rl = LogisticRegression(
            C=1.0, solver="liblinear", class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        )
        modelo_rl.fit(X_train, y_train)
        mejores_modelos["Regresión Logística"] = modelo_rl

    return mejores_modelos


def metrics_from_models(mejores_modelos, y_test, X_test) -> tuple[list[dict], list[dict]]:
    comparativa, detallada = [], []
    for nombre, modelo in mejores_modelos.items():
        y_pred = modelo.predict(X_test)
        y_proba = modelo.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, y_proba))
        rec = float(recall_score(y_test, y_pred))
        gini = 2 * auc - 1
        comparativa.append({"Modelo": nombre, "ROC-AUC": auc, "Gini": gini, "Recall": rec})
        detallada.append({
            "Modelo": nombre,
            "Accuracy": float(accuracy_score(y_test, y_pred)),
            "Precision": float(precision_score(y_test, y_pred)),
            "Recall": rec,
            "F1-Score": float(f1_score(y_test, y_pred)),
            "ROC-AUC": auc,
            "Gini (Test)": gini,
        })
    comparativa.sort(key=lambda x: x["ROC-AUC"], reverse=True)
    detallada.sort(key=lambda x: x["ROC-AUC"], reverse=True)
    return comparativa, detallada


def confusion_summary(modelo, nombre: str, y_test, X_test) -> dict:
    y_pred = modelo.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return {
        "modelo": nombre,
        "umbral": UMBRAL,
        "matriz": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "impacto": {
            "clientes_buenos_detectados": int(tn),
            "morosos_evitados": int(tp),
            "morosos_escapados": int(fn),
            "buenos_rechazados_error": int(fp),
        },
        "test_size": int(len(y_test)),
    }


def save_outputs(payload: dict) -> None:
    METRICAS_DIR.mkdir(exist_ok=True)
    slug = payload["metadata"]["slug"]
    historico_json = METRICAS_DIR / f"{slug}.json"
    body = json.dumps(payload, indent=2, ensure_ascii=False)

    historico_json.write_text(body, encoding="utf-8")
    log(f"Historico: {historico_json}")
    log(f"Vista consolidada: {METRICAS_DIR / 'README.md'}")


def main() -> int:
    t_start = time.time()
    df, num_cols, cat_cols = load_and_prepare()

    if "LGD_Real" in num_cols:
        num_cols.remove("LGD_Real")
    predictores = num_cols + cat_cols
    X_raw = df[predictores]
    y = df["target_moroso"]

    log("Train/test split...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_train = pd.get_dummies(X_train_raw, columns=cat_cols, drop_first=True)
    X_test = pd.get_dummies(X_test_raw, columns=cat_cols, drop_first=True)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    log(f"Train: {len(X_train):,} | Test: {len(X_test):,} | Features: {X_train.shape[1]}")

    mejores_modelos = train_models(X_train, X_test, y_train, y_test)
    comparativa, detallada = metrics_from_models(mejores_modelos, y_test, X_test)
    modelos_apto = sum(1 for r in detallada if r["Gini (Test)"] > 0.45)

    if CHAMPION_CM not in mejores_modelos:
        log(f"AVISO: {CHAMPION_CM} no encontrado; usando primer modelo.")
        champion_name = next(iter(mejores_modelos))
    else:
        champion_name = CHAMPION_CM
    cm_payload = confusion_summary(mejores_modelos[champion_name], champion_name, y_test, X_test)

    slug = output_slug()
    payload = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "slug": slug,
            "porcentaje_datos": PORCENTAJE_DATOS,
            "modo_completo": MODO_COMPLETO,
            "eliminar_cols_redundantes": ELIMINAR_COLS_REDUNDANTES,
            "filas_total": int(len(df)),
            "filas_train": int(len(X_train)),
            "filas_test": int(len(X_test)),
            "features": int(X_train.shape[1]),
            "duracion_segundos": round(time.time() - t_start, 1),
            "notas": [
                "flags en vars_tiempo + fillna(0)",
                "cols_redundantes eliminadas" if ELIMINAR_COLS_REDUNDANTES else "cols_redundantes conservadas",
                "solo Regresión Logística" if not MODO_COMPLETO else "4 modelos con RandomizedSearchCV",
                "sin variables leakage",
                "titulo_empleo eliminado",
            ],
        },
        "comparativa_final": comparativa,
        "ranking_robustez": detallada,
        "metricas_detalladas": detallada,
        "modelos_gini_apto": modelos_apto,
        "matriz_confusion_champion": cm_payload,
    }
    save_outputs(payload)
    log(f"Completado en {payload['metadata']['duracion_segundos']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
