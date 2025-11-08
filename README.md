# 🧩 Unificador Corporativo PRO – Streamlit  
🚀 Limpia, normaliza y unifica datos empresariales automáticamente.

---

## ✅ ¿Qué hace esta aplicación?
Esta herramienta permite:

- ✅ Unificar nombres de empresas automáticamente  
- ✅ Corregir errores comunes (tildes, espacios, S.A.S / SAS, LTDA, etc.)  
- ✅ Detectar coincidencias usando inteligencia fuzzy  
- ✅ Calcular AMOUNT, VAT y FINALAMOUNT sin errores  
- ✅ Generar reportes finales listos para contabilidad  
- ✅ Exportar resultados en Excel  
- ✅ Registrar auditoría de cada ejecución  

---

## ✅ Usar la app online (sin instalar nada)
Puedes usar la app directamente desde internet:

👉 **https://unifierapp-tdczlaniwwc8nykrgfvnv.streamlit.app**

---

## ✅ ¿Cómo usarla en tu PC? (explicado para principiantes)

### ✅ 1. Instalar Python  
Descargar desde:  
https://www.python.org/downloads/

---

### ✅ 2. Crear entorno virtual  
```bash
python -m venv venv

✅ 3. Activar entorno virtual

En Windows:

venv\Scripts\activate

✅ 4. Instalar dependencias
pip install -r requirements.txt

streamlit run app.py

unifier_app/
│── app.py
│── requirements.txt
│── logs_auditoria.csv   (se genera solo)
│── .gitignore
│── README.md
└── venv/ (opcional, ignorado por Git)


✅ Tecnologías usadas

Python 3

Streamlit

Pandas

RapidFuzz

OpenPyXL

Unidecode

✅ Autor

👤 Jhon Mario Padilla Rojas
📧 jmpadillar.7@gmail.com

GitHub: https://github.com/Yato6od