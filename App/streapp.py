import streamlit as st
import sys
import os
from datetime import datetime

# Configuración de rutas
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "modules"))

try:
    from modules.campesino import view_campesino
    from modules.transportista import view_transportista
    from modules.comprador import view_comprador
except ImportError as e:
    st.error(f"Error al importar módulos: {e}")
    st.info("Asegúrate de que existan los archivos en la carpeta 'modules/'")

# Configuración de la página
st.set_page_config(
    page_title="AgroMove IA",
    page_icon="🚜",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Pacifico&family=Poppins:wght@400;600&display=swap');
    
        .titulo-agromove {
        font-family: 'Pacifico', cursive;
        font-size: 56px;
        text-align: center;
        margin-bottom: 0px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        
        
        background: linear-gradient(270deg, #2E8B57, #3CB371, #66CDAA, #8FBC8F, #2E8B57, #3CB371, #66CDAA);
        background-size: 200% 200%; /* tamaño más grande para permitir el movimiento */
        
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        
        animation: colorShift 8s linear infinite; /* velocidad constante */
    }

    @keyframes colorShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }



    .subtitulo-agromove {
        font-family: 'Poppins', sans-serif;
        font-size: 20px;
        text-align: center;
        color: #444;
        margin-top: -10px;
        letter-spacing: 1px;
        margin-bottom: 30px;
    }

    .stButton>button {
        background-color: #2E8B57;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        border: none;
        transition: all 0.3s;
    }

    .stButton>button:hover {
        background-color: #246B45;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }

    [data-testid="stSidebar"] {
        background-color: #f8fdf8;
    }

    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: #f0f8f0;
        padding: 10px;
        text-align: center;
        font-size: 12px;
        color: #666;
        border-top: 1px solid #ddd;
    }
    </style>
""", unsafe_allow_html=True)

# Título principal con animación
st.markdown("""
    <h1 class="titulo-agromove">🚜 AgroMov</h1>
    <p class="subtitulo-agromove">Plataforma rural logística con inteligencia artificial</p>
""", unsafe_allow_html=True)

# Sidebar sin autenticación
with st.sidebar:
    st.title("🌾 Menú Principal")

    rol = st.selectbox(
        "Selecciona tu rol",
        ["Seleccionar...", "Campesino", "Transportista", "Comprador", "Administrador"],
        index=0
    )

   
    st.markdown("---")
    st.markdown("### ℹ️ Información")
    st.caption(f"Fecha: {datetime.now().strftime('%d/%m/%Y')}")
    st.caption("Versión: 1.0.0")
    st.caption("© 2025 AgroMov")

# Contenido principal
if rol == "Seleccionar...":
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 🌾 Campesinos
        - Publica tus productos
        - Gestiona inventario
        - Conecta con transportistas
        - Recibe pagos seguros
        """)

    with col2:
        st.markdown("""
        ### 🚚 Transportistas
        - Encuentra cargas disponibles
        - Optimiza rutas con IA
        - Gestiona entregas
        - Aumenta ingresos
        """)

    with col3:
        st.markdown("""
        ### 🏪 Compradores
        - Accede a productos frescos
        - Compra directa al productor
        - Precios justos
        - Trazabilidad completa
        """)

    st.markdown("---")
    

elif rol == "Campesino":
    try:
        view_campesino()
    except Exception as e:
        st.error(f"Error al cargar módulo de campesino: {e}")
        st.code("Crea el archivo: modules/campesino.py")

elif rol == "Transportista":
    try:
        view_transportista()
    except Exception as e:
        st.error(f"Error al cargar módulo de transportista: {e}")
        st.code("Crea el archivo: modules/transportista.py")

elif rol == "Comprador":
    try:
        view_comprador()
    except Exception as e:
        st.error(f"Error al cargar módulo de comprador: {e}")
        st.info("Interfaz para compradores en desarrollo...")
        st.header("🏪 Panel de Comprador")

        tab1, tab2, tab3 = st.tabs(["📦 Productos", "🛒 Mis Pedidos", "📊 Estadísticas"])

        with tab1:
            st.subheader("Productos Disponibles")
            st.write("Aquí verás el catálogo de productos disponibles...")

        with tab2:
            st.subheader("Mis Pedidos")
            st.write("Historial de pedidos y seguimiento...")

        with tab3:
            st.subheader("Estadísticas de Compra")
            st.write("Análisis de tus compras...")

elif rol == "Administrador":
    st.header("⚙️ Panel de Administración")

    tab1, tab2, tab3, tab4 = st.tabs(["👥 Usuarios", "📊 Dashboard", "⚙️ Configuración", "📝 Logs"])

    with tab1:
        st.subheader("Gestión de Usuarios")
        st.write("Administra campesinos, transportistas y compradores...")

    with tab2:
        st.subheader("Dashboard General")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Usuarios", "234", "+12%")
        with col2:
            st.metric("Entregas Activas", "45", "+5")
        with col3:
            st.metric("Ingresos Mes", "$12,450", "+23%")
        with col4:
            st.metric("Satisfacción", "4.8/5", "+0.2")

    with tab3:
        st.subheader("Configuración del Sistema")
        st.write("Parámetros generales de la plataforma...")

    with tab4:
        st.subheader("Registro de Actividad")
        st.write("Logs del sistema y auditoría...")
