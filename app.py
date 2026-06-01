from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, HTTPException, Response, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Literal, Dict, Optional
from datetime import date, datetime, timedelta, timezone
from dateutil.parser import isoparse
import sqlite3
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os
import html

# OpenAI (async client) - Con manejo de errores mejorado
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  OpenAI no está disponible. Ejecuta: pip install openai")

# ---------- Config ----------
DB = "kakebo.db"
CATS = ["Necesidades", "Opcionales", "Cultura", "Imprevistos", "Ahorro"]
Tipo = Literal["INGRESO", "GASTO"]
Categoria = Literal["Necesidades", "Opcionales", "Cultura", "Imprevistos", "Ahorro"]

# Configuración OpenAI - VERIFICACIÓN MEJORADA
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Debug: Verificar qué está pasando con la API key
print(f"🔑 OPENAI_API_KEY existe: {OPENAI_API_KEY is not None}")
print(f"📏 Longitud API KEY: {len(OPENAI_API_KEY) if OPENAI_API_KEY else 0}")

# Cliente OpenAI solo si está disponible y hay API key VÁLIDA
client = None
if OPENAI_AVAILABLE and OPENAI_API_KEY and OPENAI_API_KEY.strip():
    if OPENAI_API_KEY.startswith('sk-'):
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        print("✅ OpenAI client inicializado correctamente")
    else:
        print("❌ OPENAI_API_KEY no tiene formato válido (debe empezar con 'sk-')")
else:
    print("❌ OpenAI client NO inicializado. Razones:")
    if not OPENAI_AVAILABLE:
        print("  - Librería OpenAI no instalada")
    if not OPENAI_API_KEY:
        print("  - OPENAI_API_KEY no encontrada en .env")
    elif not OPENAI_API_KEY.strip():
        print("  - OPENAI_API_KEY está vacía")
    elif not OPENAI_API_KEY.startswith('sk-'):
        print("  - OPENAI_API_KEY no tiene formato válido")

app = FastAPI(title="Kakebo API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Modelos ----------
class MovimientoIn(BaseModel):
    tipo: Tipo
    categoria: Categoria
    monto: float = Field(gt=0)
    fecha: date
    descripcion: str = ""

class Movimiento(MovimientoIn):
    id: int

class ResumenSemana(BaseModel):
    lunes: date
    domingo: date
    ingresos: float
    gastos: float
    balance: float
    por_categoria: Dict[str, float]

class ChatRequest(BaseModel):
    message: str
    session_id: str
    timestamp: str

class ChatResponse(BaseModel):
    reply: str
    analysis: Optional[str] = None

class MovimientoResumen(BaseModel):
    tipo: Tipo
    categoria: str
    monto: float
    fecha: str
    descripcion: str

class FinancialData(BaseModel):
    total_ingresos: float
    total_gastos: float
    balance: float
    gastos_por_categoria: Dict[str, float]
    movimientos_recientes: List[MovimientoResumen]
    categorias_gastos: List[str]
    movimientos_count: int

# ---------- DB util ----------
def init_db():
    with sqlite3.connect(DB) as con:
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA busy_timeout=5000;")
        con.execute("""
        CREATE TABLE IF NOT EXISTS movimientos(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tipo TEXT NOT NULL CHECK(tipo IN ('INGRESO','GASTO')),
            categoria TEXT NOT NULL,
            monto REAL NOT NULL,
            fecha TEXT NOT NULL,
            descripcion TEXT NOT NULL
        )
        """)

def row_to_mov(row) -> Movimiento:
    return Movimiento(
        id=row[0], tipo=row[1], categoria=row[2],
        monto=row[3], fecha=isoparse(row[4]).date(), descripcion=row[5]
    )

init_db()

# ---------- Helpers ----------
def monday_of_week(d: date) -> date:
    return d - timedelta(days=(d.isoweekday() - 1))

def domingo_of(lunes: date) -> date:
    return lunes + timedelta(days=6)

def obtener_datos_financieros() -> FinancialData:
    """Obtiene y calcula métricas financieras desde la base de datos"""
    with sqlite3.connect(DB) as con:
        cur = con.cursor()
        cur.execute("""
            SELECT tipo, categoria, monto, fecha, descripcion
            FROM movimientos
            ORDER BY date(fecha) DESC, rowid DESC
        """)
        rows = cur.fetchall()

    total_ingresos = sum(row[2] for row in rows if row[0] == "INGRESO")
    total_gastos = sum(row[2] for row in rows if row[0] == "GASTO")
    balance = total_ingresos - total_gastos

    gastos_por_categoria: Dict[str, float] = {}
    for tipo, categoria, monto, _, _ in rows:
        if tipo == "GASTO":
            gastos_por_categoria[categoria] = gastos_por_categoria.get(categoria, 0.0) + monto

    movimientos_recientes: List[MovimientoResumen] = []
    for tipo, categoria, monto, fecha_str, desc in rows[:10]:
        movimientos_recientes.append(
            MovimientoResumen(
                tipo=tipo, categoria=categoria, monto=monto, fecha=fecha_str, descripcion=desc
            )
        )

    return FinancialData(
        total_ingresos=total_ingresos,
        total_gastos=total_gastos,
        balance=balance,
        gastos_por_categoria=gastos_por_categoria,
        movimientos_recientes=movimientos_recientes,
        categorias_gastos=list(gastos_por_categoria.keys()),
        movimientos_count=len(rows),
    )

# ---------- Endpoints ----------
@app.get("/hello")
def hello():
    return {"message": "Hola desde FastAPI (Python) — Kakebo"}

@app.get("/categorias", response_model=List[str])
def categorias():
    return CATS

@app.get("/movimientos", response_model=List[Movimiento])
def listar_movs():
    with sqlite3.connect(DB) as con:
        cur = con.cursor()
        cur.execute("""
            SELECT id, tipo, categoria, monto, fecha, descripcion
            FROM movimientos
            ORDER BY date(fecha) ASC, id ASC
        """)
        out = [row_to_mov(r) for r in cur.fetchall()]
    return out

@app.post("/movimientos", response_model=Movimiento, status_code=201)
def crear_mov(m: MovimientoIn):
    try:
        with sqlite3.connect(DB) as con:
            cur = con.cursor()
            cur.execute("""
                INSERT INTO movimientos(tipo, categoria, monto, fecha, descripcion)
                VALUES(?,?,?,?,?)
            """, (m.tipo, m.categoria, float(m.monto), m.fecha.isoformat(), m.descripcion))
            new_id = cur.lastrowid
            cur.execute("""
                SELECT id, tipo, categoria, monto, fecha, descripcion
                FROM movimientos
                WHERE id=?
            """, (new_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(500, "No se pudo recuperar el movimiento recién creado")
            return row_to_mov(row)
    except sqlite3.IntegrityError as e:
        raise HTTPException(400, f"Error de integridad: {e}")

@app.get("/reportes/semana", response_model=ResumenSemana)
def reporte_semanal(ref: Optional[date] = Query(default=None, description="Fecha de referencia (YYYY-MM-DD)")):
    if ref is None:
        ref = date.today()
    lunes = monday_of_week(ref)
    domingo = domingo_of(lunes)
    with sqlite3.connect(DB) as con:
        cur = con.cursor()
        cur.execute("""
            SELECT id,tipo,categoria,monto,fecha,descripcion
            FROM movimientos
            WHERE date(fecha) BETWEEN ? AND ?
            ORDER BY date(fecha) ASC
        """, (lunes.isoformat(), domingo.isoformat()))
        rows = cur.fetchall()

    ingresos = sum(r[3] for r in rows if r[1] == "INGRESO")
    gastos = sum(r[3] for r in rows if r[1] == "GASTO")
    por_cat = {c: 0.0 for c in CATS}
    for r in rows:
        if r[1] == "GASTO" and r[2] in por_cat:
            por_cat[r[2]] += r[3]

    return ResumenSemana(
        lunes=lunes, domingo=domingo,
        ingresos=ingresos, gastos=gestos, balance=ingresos - gastos,
        por_categoria=por_cat
    )

@app.get("/reportes/semana/pdf")
def reporte_semanal_pdf(ref: Optional[date] = Query(default=None, description="Fecha de referencia (YYYY-MM-DD)")):
    if ref is None:
        ref = date.today()
    # Datos
    resumen = reporte_semanal(ref)
    with sqlite3.connect(DB) as con:
        cur = con.cursor()
        cur.execute("""
            SELECT tipo, categoria, monto, fecha, descripcion
            FROM movimientos
            WHERE date(fecha) BETWEEN ? AND ?
            ORDER BY date(fecha) ASC
        """, (resumen.lunes.isoformat(), resumen.domingo.isoformat()))
        items = cur.fetchall()

    # PDF en memoria
    buff = BytesIO()
    c = canvas.Canvas(buff, pagesize=letter)
    W, H = letter
    x, y = 50, H - 50

    # Título
    c.setFont("Helvetica-Bold", 16)
    c.drawString(x, y, f"Kakebo — Reporte semanal ({resumen.lunes} a {resumen.domingo})")
    y -= 24

    # Resumen
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Resumen")
    y -= 16
    c.setFont("Helvetica", 12)
    c.drawString(x, y, f"Ingresos: ${resumen.ingresos:,.2f}"); y -= 14
    c.drawString(x, y, f"Gastos:   ${resumen.gastos:,.2f}");   y -= 14
    c.drawString(x, y, f"Balance:  ${resumen.balance:,.2f}");  y -= 20

    # Categorías
    c.setFont("Helvetica-Bold", 12); c.drawString(x, y, "Gastos por categoría"); y -= 16
    c.setFont("Helvetica", 12)
    for cat, val in resumen.por_categoria.items():
        c.drawString(x, y, f"{cat:<12} ${val:,.2f}"); y -= 14
        if y < 80:
            c.showPage(); y = H - 50

    # Movimientos
    if y < 140:
        c.showPage(); y = H - 50
    c.setFont("Helvetica-Bold", 12); c.drawString(x, y, "Movimientos"); y -= 16
    c.setFont("Helvetica", 10)
    c.drawString(x, y, "Fecha       Tipo     Categoría        Monto        Descripción"); y -= 12
    c.drawString(x, y, "-" * 95); y -= 12
    for t, cat, m, fd, desc in items:
        line = f"{fd:12} {t:8} {cat:15} ${m:10,.2f}  {desc[:50]}"
        c.drawString(x, y, line); y -= 12
        if y < 60:
            c.showPage(); y = H - 50
    c.showPage(); c.save()

    pdf = buff.getvalue()
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="kakebo_reporte_{resumen.lunes}.pdf"'}
    )

# ---------- Endpoints para el Chat con IA ----------
@app.post("/api/ai/chat", response_model=ChatResponse)
async def chat_with_ai(request: ChatRequest):
    """
    Endpoint para el chat con IA que analiza los datos financieros
    """
    try:
        # Obtener datos financieros actuales
        financial_data = obtener_datos_financieros()

        # Verificar si OpenAI está disponible Y FUNCIONANDO
        if not client:
            reply = generar_respuesta_fallback(financial_data, request.message)
            return ChatResponse(
                reply=reply,
                analysis="Análisis básico (OpenAI no disponible)"
            )

        # Preparar el contexto para OpenAI
        gastos_lines = "\n".join(
            f"  - {categoria}: ${monto:,.2f}"
            for categoria, monto in financial_data.gastos_por_categoria.items()
        )
        
        financial_context = f"""
DATOS FINANCIEROS ACTUALES:

RESUMEN:
- Ingresos: ${financial_data.total_ingresos:,.2f}
- Gastos: ${financial_data.total_gastos:,.2f}
- Balance: ${financial_data.balance:,.2f}
- Movimientos: {financial_data.movimientos_count}

GASTOS POR CATEGORÍA:
{gastos_lines if gastos_lines else "  - Sin gastos registrados"}

PREGUNTA: {request.message}
        """.strip()

        # System prompt MÁS CONCISO
        system_prompt = f"""
Eres KakeboCoach, coach financiero especializado en el método Kakebo.

REGLAS:
- Analiza los datos REALES proporcionados
- Usa categorías: {', '.join(CATS)}
- Sé CONCISO (máximo 150 palabras)
- Da 2-3 recomendaciones ACCIONABLES
- Enfócate en control y ahorro
- Responde en español

EJEMPLO:
"Balance positivo de $X. 
• Reduce gastos en [categoría] 
• Aumenta ahorro en Y% 
• Revisa [área específica]"
        """.strip()

        # Llamar a OpenAI con timeout más corto
        try:
            print("🔄 Enviando solicitud a OpenAI...")
            response = await client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": financial_context}
                ],
                max_tokens=400,  # Más corto
                temperature=0.7,
                timeout=15.0,   # Timeout más corto
            )
            reply = response.choices[0].message.content
            print("✅ Respuesta recibida de OpenAI")
            
        except Exception as openai_error:
            print(f"❌ Error con OpenAI: {openai_error}")
            reply = generar_respuesta_fallback(financial_data, request.message)

        return ChatResponse(
            reply=reply,
            analysis="Análisis completado con IA"
        )

    except Exception as e:
        print(f"💥 Error general en chat: {e}")
        raise HTTPException(status_code=500, detail=f"Error en el análisis: {str(e)}")

def generar_respuesta_fallback(financial_data: FinancialData, pregunta: str) -> str:
    """
    Genera una respuesta de fallback cuando OpenAI no está disponible
    """
    balance = financial_data.balance
    total_ingresos = financial_data.total_ingresos
    total_gastos = financial_data.total_gastos

    # Análisis básico del balance
    if balance > 0:
        situacion = f"✅ Balance POSITIVO: ${balance:,.2f}"
        recomendacion = "Considera invertir o crear fondo de emergencia."
    else:
        situacion = f"⚠️ Balance NEGATIVO: -${abs(balance):,.2f}"
        recomendacion = "Enfócate en reducir gastos no esenciales."

    # Análisis de categorías de gasto
    analisis_categorias = ""
    if financial_data.gastos_por_categoria:
        mayor_gasto = max(financial_data.gastos_por_categoria.items(), key=lambda x: x[1])
        analisis_categorias = f"Mayor gasto: {mayor_gasto[0]} (${mayor_gasto[1]:,.2f})"

    # Respuesta basada en palabras clave de la pregunta
    pregunta_lower = pregunta.lower()

    if any(palabra in pregunta_lower for palabra in ["resumen", "general", "cómo estoy", "como estoy"]):
        respuesta = f"""
{situacion}

• Ingresos: ${total_ingresos:,.2f}
• Gastos: ${total_gastos:,.2f}
• Movimientos: {financial_data.movimientos_count}

{analisis_categorias}
{recomendacion}
        """.strip()

    elif any(palabra in pregunta_lower for palabra in ["gasto", "gastar", "reducir"]):
        primera_cat = next(iter(financial_data.gastos_por_categoria.keys()), "tus categorías")
        respuesta = f"""
{analisis_categorias}

Consejos:
• Revisa gastos en {primera_cat}
• Establece límites semanales
• Prioriza Necesidades sobre Opcionales

{situacion}
        """.strip()

    elif any(palabra in pregunta_lower for palabra in ["ahorro", "ahorrar", "invertir"]):
        respuesta = f"""
{situacion}

Estrategias:
• Destina 20% al ahorro (${total_ingresos * 0.2:,.2f}/mes)
• Usa categoría 'Ahorro'
• Automatiza tus ahorros
        """.strip()

    else:
        respuesta = f"""
{situacion}

• Ingresos: ${total_ingresos:,.2f}
• Gastos: ${total_gastos:,.2f}
{analisis_categorias}

{recomendacion}
        """.strip()

    return respuesta

# ---------- Panel Web ----------
def generate_panel_html():
    with sqlite3.connect(DB) as con:
        cur = con.cursor()
        cur.execute("""
            SELECT fecha, tipo, categoria, monto, descripcion
            FROM movimientos
            ORDER BY date(fecha) ASC, rowid ASC
        """)
        rows = cur.fetchall()

    if not rows:
        body = '<tr><td colspan="5">Sin datos</td></tr>'
    else:
        body = "".join(
            f"<tr>"
            f"<td>{html.escape(f)}</td>"
            f"<td>{html.escape(t)}</td>"
            f"<td>{html.escape(c)}</td>"
            f"<td>{m:.2f}</td>"
            f"<td>{html.escape(d)}</td>"
            f"</tr>"
            for (f, t, c, m, d) in rows
        )

    return f"""
    <html><head><meta charset="utf-8"><title>Kakebo Panel</title>
    <style>
        body{{font-family:system-ui; margin:20px}}
        table{{border-collapse:collapse; width:100%; margin-top:20px}}
        td,th{{border:1px solid #ddd; padding:8px; text-align:left}}
        th{{background:#f3f3f3; font-weight:bold}}
        tr:nth-child(even){{background:#f9f9f9}}
        .header{{display:flex; justify-content:space-between; align-items:center}}
        code{{background:#f1f1f1; padding:2px 6px; border-radius:4px}}
    </style></head>
    <body>
      <div class="header">
        <h2>Movimientos Kakebo</h2>
        <small>Accedido desde: /panel</small>
      </div>
      <table>
        <thead><tr><th>Fecha</th><th>Tipo</th><th>Categoría</th><th>Monto</th><th>Descripción</th></tr></thead>
        <tbody>{body}</tbody>
      </table>
    </body></html>
    """

@app.get("/panel", response_class=HTMLResponse)
def panel():
    return generate_panel_html()

@app.get("/pane", response_class=HTMLResponse)
@app.get("/Pane", response_class=HTMLResponse)
@app.get("/Panel", response_class=HTMLResponse)
@app.get("/PANEL", response_class=HTMLResponse)
@app.get("/panal", response_class=HTMLResponse)
@app.get("/painel", response_class=HTMLResponse)
def panel_variations(request: Request):
    print(f"🔍 Acceso a variante: {request.url.path}")
    html_content = generate_panel_html()
    html_content = html_content.replace(
        "Accedido desde: /panel",
        f"Accedido desde: <code>{html.escape(request.url.path)}</code>"
    )
    return HTMLResponse(content=html_content)

# ---------- Health Check ----------
@app.get("/health")
def health_check():
    try:
        with sqlite3.connect(DB) as con:
            con.execute("SELECT 1")
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {e}"
    
    openai_status = "available" if client else "unavailable"
    
    return {
        "status": "healthy" if db_status == "connected" else "degraded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database": db_status,
        "openai": openai_status,
        "version": "1.0.0"
    }

# ---------- Run local ----------
if __name__ == "__main__":
    import uvicorn
    print("🚀 Iniciando servidor Kakebo...")
    print("📊 Disponible en: http://localhost:8000")
    print("📚 Documentación: http://localhost:8000/docs")
    print("💬 Chat IA: http://localhost:8000/api/ai/chat")
    print("🛑 Presiona Ctrl+C para detener\n")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
