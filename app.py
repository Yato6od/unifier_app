import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import re
from datetime import datetime
from unidecode import unidecode

# --------------------- Fuzzy engine ---------------------
try:
    from rapidfuzz import fuzz, process
    FUZZ_ENGINE = "rapidfuzz"
except Exception:
    from fuzzywuzzy import fuzz, process
    FUZZ_ENGINE = "fuzzywuzzy"

# --------------------- UI base --------------------------
st.set_page_config(page_title="Unificador Corporativo PRO", layout="wide")
st.markdown("""
<style>
.stApp{background:#0e1117!important;color:#e6eef8!important;}
h1,h2,h3{color:#9ad0ff!important;}
.stButton>button{background:#1f6feb!important;color:#fff!important;border-radius:10px!important;font-weight:700!important}
.stButton>button:hover{background:#1857b8!important}
.scrollbtn{position:fixed;right:18px;width:40px;height:40px;background:#1f6feb;color:#fff;
border-radius:50%;display:flex;align-items:center;justify-content:center;cursor:pointer;z-index:9999;}
#topbtn{bottom:80px;}#botbtn{bottom:25px;}
.alert-ok{color:#16c47f;font-weight:700}
.alert-warn{color:#ffcc00;font-weight:700}
.alert-bad{color:#ff6b6b;font-weight:700}
.card{background:#121826;border:1px solid #1f2633;border-radius:12px;padding:14px;margin-bottom:10px}
.small{font-size:0.92rem;opacity:.9}
</style>
<div id="topbtn" class="scrollbtn" onclick="window.scrollTo({top:0,behavior:'smooth'})">⬆</div>
<div id="botbtn" class="scrollbtn" onclick="window.scrollTo({top:document.body.scrollHeight,behavior:'smooth'})">⬇</div>
""", unsafe_allow_html=True)

# --------------------- Helpers --------------------------
def detect_column(columns, patterns):
    for p in patterns:
        for c in columns:
            if p in c.lower():
                return c
    return None

def is_numeric_str(s: str) -> bool:
    return bool(re.fullmatch(r"\d+", str(s or "").strip()))

def to_numeric(x):
    if pd.isna(x): return np.nan
    s = re.sub(r"[^\d,.\-]", "", str(x))
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        if len(s.split(",")[-1]) <= 2:
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    try: return float(s)
    except: return np.nan

def normalize_company(s):
    if pd.isna(s): return ""
    s = unidecode(str(s)).upper().strip()
    s = re.sub(r"\bS[\s\.]*A[\s\.]*S\b", "S.A.S", s)
    s = re.sub(r"\bS[\s\.]*A\b", "S.A.", s)
    s = re.sub(r"\bLTDA\b", "LTDA.", s)
    s = re.sub(r"\s+", " ", s)
    return s

# --------- Robust detection for TAX and NAME -------------
def detect_buyer_tax_id(df: pd.DataFrame):
    best = None; best_score = -1
    for col in df.columns:
        cleaned = df[col].astype(str).str.replace(r"[^\d]", "", regex=True)
        lens = cleaned.str.len()
        score = ((lens.between(8,13)).mean())  # % of values 8..13 digits
        if score > best_score:
            best_score = score; best = col
    return best

def detect_buyer_name(df: pd.DataFrame, tax_col: str|None):
    candidates = []
    for col in df.columns:
        if col == tax_col: continue
        series = df[col].astype(str)
        numeric_ratio = series.apply(lambda s: is_numeric_str(re.sub(r"[^\d]", "", s))).mean()
        avg_len = series.str.len().mean()
        # prefer texty columns (less numeric, longer text)
        score = (1 - numeric_ratio) * 0.7 + (min(avg_len, 40)/40) * 0.3
        candidates.append((score, col))
    if not candidates: return None
    candidates.sort(reverse=True)
    return candidates[0][1]

# ---------------------- Instructions button -----------------------
if st.button("📘 Ver instrucciones fáciles"):
    st.session_state["show_instructions"] = True

if st.session_state.get("show_instructions", False):
    st.markdown("---")
    st.header("📘 Instrucciones súper fáciles (para todos 👶🧑‍💻)")
    st.markdown("""
1) **Sube tu archivo** Excel o CSV (como si subieras una foto).  
2) **La app entiende solita** tus columnas.  
3) **Mueve la barrita** para encontrar nombres parecidos.  
4) Toca **“Aplicar unificaciones”** para limpiar y sumar.  
5) Toca **“Descargar archivo final”**. ¡Listo!

**Cómo ejecutar sin VS Code:**

**Requisitos:** `streamlit, pandas, openpyxl, unidecode, rapidfuzz (o fuzzywuzzy), python-Levenshtein`
""")
    st.markdown("---")

# ---------------------- Title & upload -----------------------------
st.title("📊 Unificador Corporativo PRO – Modo Oscuro")
uploaded = st.file_uploader("📁 Sube tu archivo XLSX o CSV", type=["xlsx","xls","csv"])
if not uploaded: st.stop()

# ---------------------- Read file ---------------------------------
try:
    if uploaded.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded, dtype=str)
    else:
        df = pd.read_excel(uploaded, dtype=str)
except Exception as e:
    st.error(f"❌ Error al leer el archivo: {e}")
    st.stop()

df.columns = df.columns.str.upper().str.strip()
if "FUENTE" not in df.columns:
    df["FUENTE"] = "DESCONOCIDO"

st.subheader("👀 Vista previa del archivo")
st.dataframe(df.head())

# ---------------------- Auto-detect columns -----------------------
tax_col = detect_buyer_tax_id(df)
name_col = detect_buyer_name(df, tax_col)
amount_col = detect_column(df.columns, ["amount","base","subtotal","valor"])
vat_col = detect_column(df.columns, ["vat","iva","impuesto"])
final_col = detect_column(df.columns, ["final","total"])
month_col = detect_column(df.columns, ["month","mes"])

detected = {
    "FUENTE": "FUENTE",
    "BUYER_TAX_ID": tax_col,
    "BUYER_NAME": name_col,
    "AMOUNT": amount_col,
    "VAT": vat_col,
    "FINALAMOUNT": final_col,
    "MONTH": month_col,
}
st.subheader("🧠 Columnas detectadas inteligentemente")
st.write(detected)

# ---------------------- Manual override ---------------------------
with st.expander("🔧 (Opcional) Elegir columnas manualmente"):
    cols = df.columns.tolist()
    tax_col   = st.selectbox("Columna BUYER_TAX_ID (NIT)", cols, index=cols.index(tax_col) if tax_col in cols else 0)
    name_col  = st.selectbox("Columna BUYER_NAME (Razón social)", cols, index=cols.index(name_col) if name_col in cols else 0)
    amount_col= st.selectbox("Columna AMOUNT", cols, index=cols.index(amount_col) if amount_col in cols else 0)
    vat_col   = st.selectbox("Columna VAT", cols, index=cols.index(vat_col) if vat_col in cols else 0)
    final_col = st.selectbox("Columna FINALAMOUNT", cols, index=cols.index(final_col) if final_col in cols else 0)
    month_col = st.selectbox("Columna MONTH", cols, index=cols.index(month_col) if month_col in cols else 0)

# Validations of presence
needed = [name_col, amount_col, vat_col]
if any(c is None for c in needed):
    st.error("❗ Faltan columnas esenciales: BUYER_NAME, AMOUNT o VAT.")
    st.stop()

# --------------------- Controls ----------------------
sim_threshold = st.slider("Porcentaje de coincidencia (%)", 50, 100, 90)
LOG_FILE = "logs_auditoria.csv"

def log_event(data: dict):
    df_log = pd.DataFrame([data])
    if os.path.exists(LOG_FILE):
        df_log = pd.concat([pd.read_csv(LOG_FILE), df_log], ignore_index=True)
    df_log.to_csv(LOG_FILE, index=False)

# --------------------- Process button -----------------
st.markdown("---")
run_checks = st.checkbox("🔎 Revisar errores antes de exportar", value=True)
if st.button("✅ APLICAR UNIFICACIONES"):

    # -------- Normalize & numeric ----------
    df2 = df.copy()
    df2["_NORM"] = df2[name_col].apply(normalize_company)
    df2["_AMOUNT"] = df2[amount_col].apply(to_numeric)
    df2["_VAT"] = df2[vat_col].apply(to_numeric)
    df2["_FINAL_RAW"] = df2[final_col].apply(to_numeric) if final_col else np.nan
    df2["_FINAL"] = df2["_FINAL_RAW"]
    df2.loc[df2["_FINAL"].isna(), "_FINAL"] = df2["_AMOUNT"] + df2["_VAT"]

    # -------- Unify names (fuzzy groups) ----
    unique = df2["_NORM"].dropna().unique().tolist()
    reps, mapping = [], {}
    for n in unique:
        if not reps:
            reps.append(n); mapping[n]=n; continue
        match = process.extractOne(n, reps, scorer=fuzz.token_sort_ratio)
        if match and match[1] >= sim_threshold: mapping[n]=match[0]
        else: reps.append(n); mapping[n]=n
    df2["BUYER_UNIFIED"] = df2["_NORM"].map(mapping)

    # -------- Aggregated (for charts) -------
    group_cols = ["BUYER_UNIFIED"]
    if month_col: df2["_MONTH"] = df2[month_col]; group_cols.append("_MONTH")
    agg = df2.groupby(group_cols, as_index=False).agg(
        AMOUNT=("_AMOUNT","sum"), VAT=("_VAT","sum"), FINALAMOUNT=("_FINAL","sum")
    )
    agg["FINAL_CALCULADO"] = agg["AMOUNT"] + agg["VAT"]
    agg["DIFERENCIA"] = agg["FINALAMOUNT"] - agg["FINAL_CALCULADO"]
    agg["DIFERENCIA_%"] = (agg["DIFERENCIA"] / agg["FINALAMOUNT"].replace(0,np.nan))*100
    agg["DIFERENCIA_%"] = agg["DIFERENCIA_%"].fillna(0).round(2)

    # --------- Error checks (A) -------------
    errores_contables = df2.copy()
    errores_contables["CALC"] = errores_contables["_AMOUNT"] + errores_contables["_VAT"]
    errores_contables["DIFERENCIA"] = (errores_contables["_FINAL"] - errores_contables["CALC"]).round(2)
    errores_contables["COINCIDE"] = np.isclose(errores_contables["_FINAL"], errores_contables["CALC"], atol=0.01)
    errores_contables = errores_contables.loc[~errores_contables["COINCIDE"]]
    errores_contables = errores_contables[[
        "FUENTE", month_col if month_col else name_col, tax_col, name_col,
        "_AMOUNT","_VAT","_FINAL","CALC","DIFERENCIA"
    ]].rename(columns={
        month_col if month_col else name_col: "MONTH" if month_col else "BUYER_NAME",
        tax_col:"BUYER_TAX_ID",
        name_col:"BUYER_NAME",
        "_AMOUNT":"AMOUNT","_VAT":"VAT","_FINAL":"FINALAMOUNT"
    })

    # --------- NIT checks (B) ---------------
    nit_df = df2[tax_col].astype(str) if tax_col else pd.Series([],dtype=str)
    nit_invalid = pd.DataFrame()
    nit_dupes = pd.DataFrame()
    nit_shared = pd.DataFrame()

    if tax_col:
        cleaned = nit_df.str.replace(r"[^\d]","",regex=True)
        bad_mask = ~cleaned.str.len().between(8,13)
        nit_invalid = df2.loc[bad_mask, ["FUENTE", tax_col, name_col]].rename(columns={tax_col:"BUYER_TAX_ID", name_col:"BUYER_NAME"})
        dupes = cleaned[cleaned.duplicated(keep=False)]
        if not dupes.empty:
            nit_dupes = df2.loc[cleaned.isin(dupes), ["FUENTE", tax_col, name_col]].rename(columns={tax_col:"BUYER_TAX_ID", name_col:"BUYER_NAME"})
            # shared nit different names
            nit_shared = (nit_dupes.groupby(["BUYER_TAX_ID"])["BUYER_NAME"]
                          .nunique().reset_index().query("BUYER_NAME > 1"))

    # --------- Name checks (C) --------------
    names_series = df2[name_col].astype(str)
    only_digits = names_series.str.replace(r"[^\d]","",regex=True).str.len() > 0
    only_digits &= names_series.str.replace(r"\D","",regex=True).str.len() == names_series.str.len()
    very_short = names_series.str.len() <= 2
    junk = names_series.str.fullmatch(r"[\.\-\_]+")
    suspicious_idx = only_digits | very_short | junk
    nombres_sospechosos = df2.loc[suspicious_idx, ["FUENTE", name_col, tax_col if tax_col else name_col]].rename(
        columns={name_col:"BUYER_NAME", (tax_col if tax_col else name_col):"BUYER_TAX_ID" if tax_col else "SAMPLE"}
    )

    # --------- Dashboard (D) ----------------
    st.subheader("📈 Resumen tipo Power BI")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.metric("Empresas únicas", len(agg["BUYER_UNIFIED"].unique()))
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.metric("Monto total (AMOUNT)", f"{agg['AMOUNT'].sum():,.0f}")
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.metric("Diferencia total", f"{agg['DIFERENCIA'].sum():,.0f}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("#### 🏆 Top 10 empresas por AMOUNT")
    top10 = agg.groupby("BUYER_UNIFIED", as_index=False)["AMOUNT"].sum().sort_values("AMOUNT", ascending=False).head(10)
    st.bar_chart(top10.set_index("BUYER_UNIFIED"))

    if month_col:
        st.markdown("#### 📅 AMOUNT por mes")
        by_month = agg.groupby("_MONTH", as_index=False)["AMOUNT"].sum()
        st.bar_chart(by_month.set_index("_MONTH"))

    st.markdown("#### ⚠️ Empresas con mayor diferencia")
    diff10 = agg.sort_values("DIFERENCIA", ascending=False).head(10)
    st.dataframe(diff10)

    # --------- Alerts panel (E2) ------------
    st.markdown("### 🧯 Panel de alertas")
    ok_cnt = (len(errores_contables)==0) and (nit_invalid.empty and nit_shared.empty) and (nombres_sospechosos.empty)
    st.markdown(f"- ✅ <span class='alert-ok'>Contables OK</span>: {len(errores_contables)==0}", unsafe_allow_html=True)
    st.markdown(f"- ⚠️ <span class='alert-warn'>NIT inválidos</span>: {len(nit_invalid)}", unsafe_allow_html=True)
    st.markdown(f"- ⚠️ <span class='alert-warn'>NIT compartidos con distintos nombres</span>: {0 if nit_shared.empty else len(nit_shared)}", unsafe_allow_html=True)
    st.markdown(f"- ⚠️ <span class='alert-warn'>Nombres sospechosos</span>: {len(nombres_sospechosos)}", unsafe_allow_html=True)

    # Show detailed tables when checks enabled
    if run_checks:
        with st.expander("📄 Errores contables (FINALAMOUNT vs AMOUNT+VAT)"):
            if len(errores_contables)==0: st.info("Todo cuadra ✅")
            else: st.dataframe(errores_contables)
        with st.expander("📄 NIT inválidos"):
            if nit_invalid.empty: st.info("Sin NIT inválidos ✅")
            else: st.dataframe(nit_invalid)
        with st.expander("📄 NIT compartidos con diferentes nombres"):
            if nit_shared.empty: st.info("Sin conflictos de NIT ✅")
            else: st.dataframe(nit_shared)
        with st.expander("📄 Nombres sospechosos"):
            if nombres_sospechosos.empty: st.info("Sin nombres sospechosos ✅")
            else: st.dataframe(nombres_sospechosos)

    # --------- Final file (corporate columns) ------
    finalA = pd.DataFrame({
        "FUENTE": df2["FUENTE"],
        "BUYER_TAX_ID": df2[tax_col] if tax_col else "",
        "BUYER_NAME": df2["BUYER_UNIFIED"],
        "AMOUNT": df2["_AMOUNT"],
        "VAT": df2["_VAT"],
        "FINALAMOUNT": df2["_FINAL"]
    })
    if month_col: finalA.insert(1, "MONTH", df2[month_col])

    # --------- Export to Excel with sheets ----------
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as w:
        finalA.to_excel(w, index=False, sheet_name="ARCHIVO_FINAL")
        agg.to_excel(w, index=False, sheet_name="UNIFICADO")
        if len(errores_contables): errores_contables.to_excel(w, index=False, sheet_name="ERRORES_CONTABLES")
        if not nit_invalid.empty: nit_invalid.to_excel(w, index=False, sheet_name="NIT_INVALIDOS")
        if not nit_shared.empty: nit_shared.to_excel(w, index=False, sheet_name="NIT_COMPARTIDOS")
        if not nombres_sospechosos.empty: nombres_sospechosos.to_excel(w, index=False, sheet_name="NOMBRES_SOSPECHOSOS")

    buffer.seek(0)
    fname = "UNIFICADO_PRO_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".xlsx"
    st.download_button("📥 Descargar archivo final (todas las hojas)", data=buffer.getvalue(),
                       file_name=fname, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # -------------- Log audit ----------------
    log_event({
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "archivo": uploaded.name,
        "motor": FUZZ_ENGINE,
        "umbral": sim_threshold,
        "filas": len(df),
        "errores_contables": int(len(errores_contables)),
        "nit_invalidos": int(len(nit_invalid)),
        "nit_conflictos": 0 if nit_shared.empty else int(len(nit_shared)),
        "nombres_sospechosos": int(len(nombres_sospechosos)),
        "archivo_generado": fname
    })

# --------------------- Download logs -------------------
if os.path.exists("logs_auditoria.csv"):
    st.markdown("---")
    st.subheader("📜 Descargar log de auditoría")
    with open("logs_auditoria.csv","rb") as f:
        st.download_button("📥 Descargar logs_auditoria.csv", f.read(), file_name="logs_auditoria.csv", mime="text/csv")

st.markdown("---")
st.caption("Desarrollado por **Jhon Mario Padilla Rojas** — 📧 jmpadillar.7@gmail.com")
