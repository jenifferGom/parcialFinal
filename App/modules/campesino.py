import streamlit as st
import pandas as pd
import os
from datetime import datetime
from geopy.geocoders import Nominatim
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json
import numpy as np
from pathlib import Path
from PIL import Image
import time
import folium
from streamlit_folium import st_folium
from folium import plugins


# CONFIGURACIÓN DE RUTAS Y ARCHIVOS

CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_ROOT = os.path.dirname(CURRENT_FILE_DIR)

DATA_DIR = os.path.join(APP_ROOT, "data")
IMG_DIR = os.path.join(APP_ROOT, "modules", "imagenes_productos")
MODELS_DIR = os.path.join(APP_ROOT, "models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

CSV_NOTIFICACIONES = os.path.join(DATA_DIR, "notificaciones_transporte.csv")

# Configuración de auto-refresh
AUTO_REFRESH_INTERVAL = 5  # segundos


# PRESENTACIONES Y EQUIVALENCIAS (GLOBAL)


PRESENTACIONES = {
    "Plátano": ["Kilogramo (1 kg)", "Bolsa (≈50 kg)"],
    "Papa": ["Bulto (50 kg)"],
    "Yuca": ["Bolsa (≈50 kg)"],
    "Arveja": ["Bulto (50 kg)"],
    "Frijol": ["Bulto (50 kg)"],
    "Habichuela": ["Bulto (50 kg)"],
    "Ahuyama": ["Kilogramo (1 kg)", "Bulto (50 kg)"],
    "Ajo": ["Atado/Manojo (≈10 kg)", "Caja de cartón (≈20 kg)"],
    "Berenjena": ["Kilogramo (1 kg)"],
    "Calabacín": ["Bolsa (≈50 kg)"],
    "Chócolo": ["Bulto (50 kg)"],
    "Cidra": ["Kilogramo (1 kg)"],
    "Pepino": ["Kilogramo (1 kg)", "Caja de cartón (≈20 kg)", "Canastilla (≈20 kg)"],
    "Pimentón": ["Kilogramo (1 kg)"],
    "Remolacha": ["Bulto (50 kg)"],
    "Rábano": ["Atado/Manojo (≈10 kg)"],
    "Zanahoria": ["Bulto (50 kg)"],
    "Cebolla": ["Bulto (50 kg)", "Atado/Manojo (≈10 kg)"],
    "Tomate": ["Canastilla (≈20 kg)"],
    "Acelga": ["Atado/Manojo (≈10 kg)"],
    "Apio": ["Atado/Manojo (≈10 kg)"],
    "Manzana": ["Caja (≈20 kg)"],
    "Banana": ["Bolsa (≈50 kg)"],
    "Naranja": ["Unidad (≈0.2 kg)"],
    "Pera": ["Caja (≈20 kg)"]
}

KG_EQUIVALENTES = {
    "Kilogramo (1 kg)": 1,
    "Bolsa (≈50 kg)": 50,
    "Canastilla (≈20 kg)": 20,
    "Bulto (50 kg)": 50,
    "Atado/Manojo (≈10 kg)": 10,
    "Caja de cartón (≈20 kg)": 20,
    "Caja (≈20 kg)": 20,
    "Unidad (≈0.2 kg)": 0.2
}


# MAPEO DE NOMBRES DEL MODELO A PRESENTACIONES

MAPEO_NOMBRES = {
    "Papa": "Papa",
    "platano": "Plátano",
    "yuca": "Yuca",
    "arveja": "Arveja",
    "frijol": "Frijol",
    "habichuela": "Habichuela",
    "ahuyama": "Ahuyama",
    "ajo": "Ajo",
    "berenjena": "Berenjena",
    "calabacin": "Calabacín",
    "chocolo": "Chócolo",
    "cidra": "Cidra",
    "pepino": "Pepino",
    "pimenton": "Pimentón",
    "remolacha": "Remolacha",
    "rabano": "Rábano",
    "zanahoria": "Zanahoria",
    "cebolla": "Cebolla",
    "tomate": "Tomate",
    "acelga": "Acelga",
    "apio": "Apio",
    "manzana": "Manzana",
    "banana": "Banana",
    "naranja": "Naranja",
    "pera": "Pera"
}


# CARGA DE MODELOS DE IA

MODEL_PRECIO_PATH = os.path.join(MODELS_DIR, "modelo_precios.pkl")
MODEL_CALIDAD_PATH = os.path.join(MODELS_DIR, "modelo_rutas_boyaca.pkl")

modelo_precio = None
modelo_calidad = None
IA_CARGADA = False

with st.sidebar.expander("🔍 Debug Info", expanded=False):
    st.write("📂 App Root:", APP_ROOT)
    st.write("📂 Models Dir:", MODELS_DIR)
    
    if os.path.exists(MODELS_DIR):
        st.write("✅ Carpeta models/ existe")
        st.write("📋 Archivos:", os.listdir(MODELS_DIR))
    else:
        st.error("❌ Carpeta models/ no existe")
    
    try:
        if os.path.exists(MODEL_PRECIO_PATH):
            modelo_precio = joblib.load(MODEL_PRECIO_PATH)
            st.success("✅ Modelo de precios cargado")
        else:
            st.warning("⚠️ No se encontró: modelo_precios.pkl")
        
        if os.path.exists(MODEL_CALIDAD_PATH):
            modelo_calidad = joblib.load(MODEL_CALIDAD_PATH)
            st.success("✅ Modelo de rutas cargado")
        else:
            st.warning("⚠️ No se encontró: modelo_rutas_boyaca.pkl")
        
        if modelo_precio is not None or modelo_calidad is not None:
            IA_CARGADA = True
            st.success("🎉 Sistema IA activado")
        else:
            st.error("❌ No se cargó ningún modelo")
    except Exception as e:
        st.error(f"❌ Error: {e}")
        modelo_precio = None
        modelo_calidad = None
        IA_CARGADA = False


# CARGA DEL MODELO DE CLASIFICACIÓN DE IMÁGENES

MODELO_IA_PATH = os.path.join(MODELS_DIR, "modelo_papa_final.h5")
CLASES_PATH = os.path.join(MODELS_DIR, "clases_papa.json")

@st.cache_resource
def cargar_modelo_clasificacion():
    try:
        modelo = load_model(MODELO_IA_PATH)
        with open(CLASES_PATH, "r") as f:
            clases = json.load(f)
        clases_inv = {v: k for k, v in clases.items()}
        return modelo, clases_inv
    except Exception as e:
        st.error(f"❌ Error cargando modelo de clasificación: {e}")
        return None, {}

modelo_clasificacion, clases_inv = cargar_modelo_clasificacion()


# INICIALIZACIÓN DE SESSION STATE

def inicializar_session_state():
    """Inicializa variables de session state"""
    if 'refresh_counter' not in st.session_state:
        st.session_state.refresh_counter = 0
    if 'auto_refresh_enabled' not in st.session_state:
        st.session_state.auto_refresh_enabled = True


# FUNCIONES AUXILIARES

def cargar_notificaciones():
    if os.path.exists(CSV_NOTIFICACIONES):
        df = pd.read_csv(CSV_NOTIFICACIONES)
        columnas_requeridas = {
            'transportista_lat': None,
            'transportista_lon': None,
            'distancia_restante_km': None,
            'progreso_viaje': 0.0,
            'tiempo_estimado_llegada': None,
            'ruta_optimizada': None,
            'orden_parada': None
        }
        for col, default in columnas_requeridas.items():
            if col not in df.columns:
                df[col] = default
        return df
    else:
        return pd.DataFrame(columns=[
            'id_notificacion', 'fecha_notificacion', 'campesino', 'producto',
            'cantidad_kg', 'ciudad', 'direccion', 'precio', 'precio_predicho',
            'calidad', 'estado', 'fecha_recogida', 'transportista_asignado',
            'imagen', 'latitud', 'longitud', 'transportista_lat', 'transportista_lon',
            'distancia_restante_km', 'progreso_viaje', 'tiempo_estimado_llegada',
            'ruta_optimizada', 'orden_parada'
        ])

def guardar_notificaciones(df):
    df.to_csv(CSV_NOTIFICACIONES, index=False)

def guardar_imagen_subida(archivo_subido, prefix="imagen"):
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    extension = archivo_subido.name.split('.')[-1]
    safe_prefix = "".join(c for c in prefix if c.isalnum() or c in ("_", "-"))[:30]
    nombre_archivo = f"{safe_prefix}_{timestamp}.{extension}"
    ruta_completa = os.path.join(IMG_DIR, nombre_archivo)
    with open(ruta_completa, "wb") as f:
        f.write(archivo_subido.getbuffer())
    return nombre_archivo

def calcular_distancia_haversine(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * 
         np.cos(np.radians(lat2)) * np.sin(dlon/2)**2)
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def obtener_coordenadas(ciudad, direccion):
    try:
        geolocator = Nominatim(user_agent="agrimarket_app")
        location = geolocator.geocode(f"{direccion}, {ciudad}, Boyacá, Colombia")
        if location:
            return location.latitude, location.longitude
    except:
        pass
    return None, None

def validar_coordenadas(lat, lon):
    """Valida que las coordenadas sean válidas"""
    if lat is None or lon is None:
        return False
    if pd.isna(lat) or pd.isna(lon):
        return False
    return True

def normalizar_nombre_producto(producto_raw):
    """Normaliza el nombre del producto detectado por IA"""
    producto_limpio = producto_raw.lower().strip().replace("_", " ")
    
    if producto_limpio in MAPEO_NOMBRES:
        return MAPEO_NOMBRES[producto_limpio]
    
    producto_capitalizado = producto_limpio.capitalize()
    
    if producto_capitalizado in PRESENTACIONES:
        return producto_capitalizado
    
    return producto_capitalizado

def predecir_producto_real(img_path):
    """Predice el producto usando el modelo de clasificación"""
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = modelo_clasificacion.predict(img_array)
        idx = int(np.argmax(pred[0]))
        confianza = float(np.max(pred[0]))
        
        producto_raw = clases_inv.get(idx, "Desconocido")
        producto = normalizar_nombre_producto(producto_raw)

        return producto, confianza
    except Exception as e:
        st.error(f"⚠️ Error al procesar imagen: {e}")
        return "Error", 0.0


# FUNCIÓN PARA CREAR MAPA INTERACTIVO CON FOLIUM

def crear_mapa_seguimiento_folium(lat_campesino, lon_campesino, lat_transportista=None, 
                                   lon_transportista=None, distancia_km=None, progreso=0.0,
                                   transportista_nombre=None):
    """
    Crea un mapa interactivo con Folium mostrando seguimiento en tiempo real
    """
    # Determinar el centro y zoom del mapa
    if lat_transportista and lon_transportista and validar_coordenadas(lat_transportista, lon_transportista):
        center_lat = (lat_campesino + lat_transportista) / 2
        center_lon = (lon_campesino + lon_transportista) / 2
        
        if distancia_km:
            if distancia_km < 5:
                zoom = 13
            elif distancia_km < 20:
                zoom = 11
            elif distancia_km < 50:
                zoom = 10
            else:
                zoom = 9
        else:
            zoom = 11
    else:
        center_lat = lat_campesino
        center_lon = lon_campesino
        zoom = 13
    
    # Crear mapa base con estilo Google Maps
    mapa = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles='https://mt1.google.com/vt/lyrs=r&x={x}&y={y}&z={z}',
        attr='Google',
        control_scale=True
    )
    
    # Agregar controles adicionales
    plugins.Fullscreen(
        position='topright',
        title='Pantalla completa',
        title_cancel='Salir de pantalla completa',
        force_separate_button=True
    ).add_to(mapa)
    
    plugins.LocateControl(auto_start=False).add_to(mapa)
    plugins.MeasureControl(position='bottomleft', primary_length_unit='kilometers').add_to(mapa)
    
    # Marcador del campesino (punto de entrega)
    folium.Marker(
        location=[lat_campesino, lon_campesino],
        popup=folium.Popup(
            f"""
            <div style='font-family: Arial; padding: 12px; width: 250px;'>
                <h4 style='color: #34A853; margin: 0 0 10px 0; border-bottom: 2px solid #34A853; padding-bottom: 5px;'>
                    🟢 Tu Ubicación
                </h4>
                <p style='margin: 5px 0;'><b>📍 Punto de entrega</b></p>
                <p style='margin: 5px 0; font-size: 11px; color: #666;'>
                    Coordenadas:<br>
                    Lat: {lat_campesino:.6f}<br>
                    Lon: {lon_campesino:.6f}
                </p>
                <p style='margin: 10px 0 0 0; padding: 8px; background: #E8F5E9; border-radius: 5px; font-size: 12px;'>
                    ✅ Esperando al transportista
                </p>
            </div>
            """,
            max_width=300
        ),
        tooltip="📍 Tu ubicación - Punto de entrega",
        icon=folium.Icon(
            color='green',
            icon='home',
            prefix='fa'
        )
    ).add_to(mapa)
    
    # Marcador del transportista (si existe y está en camino)
    if lat_transportista and lon_transportista and validar_coordenadas(lat_transportista, lon_transportista):
        tiempo_estimado = int(distancia_km / 50 * 60) if distancia_km else 0
        
        popup_content = f"""
            <div style='font-family: Arial; padding: 12px; width: 280px;'>
                <h4 style='color: #EA4335; margin: 0 0 10px 0; border-bottom: 2px solid #EA4335; padding-bottom: 5px;'>
                    🚛 Transportista en Camino
                </h4>
                {f"<p style='margin: 5px 0;'><b>👤 {transportista_nombre}</b></p>" if transportista_nombre else ""}
                <p style='margin: 5px 0; font-size: 11px; color: #666;'>
                    Ubicación actual:<br>
                    Lat: {lat_transportista:.6f}<br>
                    Lon: {lon_transportista:.6f}
                </p>
        """
        
        if distancia_km and progreso is not None:
            popup_content += f"""
                <div style='margin: 10px 0; padding: 10px; background: #FEE; border-radius: 5px;'>
                    <p style='margin: 3px 0; font-size: 13px;'>
                        📊 <b>Progreso:</b> {progreso*100:.1f}%
                    </p>
                    <p style='margin: 3px 0; font-size: 13px;'>
                        📏 <b>Distancia:</b> {distancia_km:.2f} km
                    </p>
                    <p style='margin: 3px 0; font-size: 13px;'>
                        ⏱️ <b>Tiempo est.:</b> ~{tiempo_estimado} min
                    </p>
                </div>
            """
        
        popup_content += "</div>"
        
        # Marcador animado del transportista
        folium.Marker(
            location=[lat_transportista, lon_transportista],
            popup=folium.Popup(popup_content, max_width=320),
            tooltip=f"🚛 Transportista - {progreso*100:.1f}% del recorrido",
            icon=folium.Icon(
                color='red',
                icon='truck',
                prefix='fa'
            )
        ).add_to(mapa)
        
        # Línea de ruta entre transportista y destino
        folium.PolyLine(
            locations=[
                [lat_transportista, lon_transportista],
                [lat_campesino, lon_campesino]
            ],
            color='#4285F4',
            weight=5,
            opacity=0.8,
            tooltip=f"Ruta restante: {distancia_km:.2f} km" if distancia_km else "Ruta estimada",
            dash_array='10, 5'
        ).add_to(mapa)
        
        # Círculo de proximidad alrededor del transportista
        folium.Circle(
            location=[lat_transportista, lon_transportista],
            radius=800,  # 800 metros
            color='#EA4335',
            fill=True,
            fillColor='#EA4335',
            fillOpacity=0.15,
            weight=2,
            opacity=0.6,
            tooltip='Área de cobertura del transportista'
        ).add_to(mapa)
        
        # Ajustar vista para mostrar ambos puntos
        mapa.fit_bounds([
            [lat_campesino, lon_campesino],
            [lat_transportista, lon_transportista]
        ], padding=[50, 50])
    
    # Círculo alrededor del punto de entrega
    folium.Circle(
        location=[lat_campesino, lon_campesino],
        radius=500,
        color='#34A853',
        fill=True,
        fillColor='#34A853',
        fillOpacity=0.1,
        weight=2,
        opacity=0.5,
        tooltip='Zona de entrega'
    ).add_to(mapa)
    
    return mapa

# ESTILOS CSS

def aplicar_estilos():
    st.markdown("""
    <style>
        .main {background: #f8f9fa;}
        .card-venta {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 25px;
            border-radius: 15px;
            color: white;
            margin: 15px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .precio-card {
            background: white;
            color: #333;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .tracking-card {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 20px;
            border-radius: 15px;
            color: white;
            margin: 15px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .tracking-live {
            background: linear-gradient(135deg, #FA8BFF 0%, #2BD2FF 52%, #2BFF88 90%);
            padding: 25px;
            border-radius: 15px;
            color: white;
            margin: 20px 0;
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { box-shadow: 0 6px 20px rgba(0,0,0,0.3); }
            50% { box-shadow: 0 8px 25px rgba(250,139,255,0.5); }
        }
        .stButton>button {
            width: 100%;
            border-radius: 10px;
            height: 50px;
            font-size: 16px;
            font-weight: bold;
        }
        .legend-item {
            display: inline-block;
            margin: 0 15px;
            padding: 8px 15px;
            background: rgba(255,255,255,0.9);
            border-radius: 8px;
            font-weight: bold;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .debug-box {
            background: #f0f0f0;
            border-left: 4px solid #667eea;
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
        }
        .status-badge {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: bold;
            margin: 5px;
        }
        .status-en-camino {
            background: #EA4335;
            color: white;
            animation: blink 1.5s infinite;
        }
        @keyframes blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
    </style>
    """, unsafe_allow_html=True)


# VISTA: PREDICCIÓN DE PRECIOS

def vista_prediccion_precios():
    st.markdown('<div class="card-venta"><h2>💰 Predicción de Precios</h2><p>Consulta el precio estimado de tus productos</p></div>', unsafe_allow_html=True)
    
    if not IA_CARGADA or modelo_precio is None:
        st.warning("⚠️ El modelo de predicción de precios no está disponible.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        producto = st.selectbox("🌾 Selecciona el producto:", list(PRESENTACIONES.keys()))
    
    with col2:
        presentaciones_disponibles = PRESENTACIONES.get(producto, ["Kilogramo (1 kg)"])
        presentacion = st.selectbox("📦 Presentación:", presentaciones_disponibles)
    
    cantidad = st.number_input("🔢 Cantidad de presentaciones:", min_value=1, max_value=1000, value=1)
    
    kg_por_unidad = KG_EQUIVALENTES[presentacion]
    kg_total = kg_por_unidad * cantidad
    
    st.info(f"⚖️ Total: **{kg_total} kg** ({cantidad} × {kg_por_unidad} kg)")
    
    if st.button("🔮 Calcular Precio", type="primary", use_container_width=True):
        with st.spinner("🧮 Calculando precio estimado..."):
            try:
                entrada = pd.DataFrame({
                    'producto': [producto],
                    'presentacion': [presentacion],
                    'ciudad': ['Tunja'],
                    'categoria': ['General'],
                    'unidades (kg)': [kg_por_unidad]
                })
                
                pred = modelo_precio.predict(entrada)
                resultado = pred[0]
                
                try:
                    precio_min_presentacion, precio_max_presentacion = sorted(resultado)
                except:
                    precio_min_presentacion = precio_max_presentacion = float(resultado)
                
                precio_min_total = precio_min_presentacion * cantidad
                precio_max_total = precio_max_presentacion * cantidad
                precio_promedio_total = (precio_min_total + precio_max_total) / 2
                
                precio_por_kg = precio_promedio_total / kg_total
                
                st.markdown("---")
                st.markdown("### 💰 Resultado de la Predicción")
                
                col_result1, col_result2, col_result3 = st.columns(3)
                
                with col_result1:
                    st.metric(
                        label="💵 Precio Estimado",
                        value=f"${precio_promedio_total:,.0f}",
                        help=f"Precio promedio para {cantidad} {presentacion}"
                    )
                
                with col_result2:
                    st.metric(
                        label="📊 Rango de Precios",
                        value=f"${precio_min_total:,.0f} - ${precio_max_total:,.0f}",
                        help="Rango mínimo y máximo estimado"
                    )
                
                with col_result3:
                    st.metric(
                        label="⚖️ Precio por Kg",
                        value=f"${precio_por_kg:,.0f}",
                        help=f"Precio unitario por kilogramo"
                    )
                
                st.markdown(f"""
                    <div class="precio-card">
                        <h4>📦 Desglose del Precio</h4>
                        <p><b>Producto:</b> {producto}</p>
                        <p><b>Presentación:</b> {presentacion} ({kg_por_unidad} kg)</p>
                        <p><b>Cantidad:</b> {cantidad} unidad(es)</p>
                        <p><b>Peso total:</b> {kg_total} kg</p>
                        <hr>
                        <p style="font-size:12px; color:#666;">
                            💡 <b>Precio por presentación:</b> ${precio_min_presentacion:,.0f} - ${precio_max_presentacion:,.0f}<br>
                            💡 <b>Precio por kg:</b> ${precio_por_kg:,.0f}<br>
                            💡 <b>Total ({cantidad} × {presentacion}):</b> ${precio_promedio_total:,.0f}
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.success("✅ Predicción completada exitosamente")
                
            except Exception as e:
                st.error(f"❌ Error al calcular el precio: {e}")


# VISTA: REGISTRO DE PRODUCTOS

def vista_registro():
    st.markdown('<div class="card-venta"><h2>📝 Registro de Productos</h2><p>Registra tus productos para la venta</p></div>', unsafe_allow_html=True)
    
    df_notif = cargar_notificaciones()
    
    with st.form("form_registro_producto"):
        col1, col2 = st.columns(2)
        
        with col1:
            campesino = st.text_input("👨‍🌾 Nombre del campesino:", value="Campesino")
            producto = st.selectbox("🌾 Producto:", list(PRESENTACIONES.keys()))
            presentacion = st.selectbox("📦 Presentación:", PRESENTACIONES.get(producto, ["Kilogramo (1 kg)"]))
        
        with col2:
            cantidad = st.number_input("🔢 Cantidad:", min_value=1, value=1)
            ciudad = st.text_input("🏙️ Ciudad:", value="Tunja")
            direccion = st.text_area("📍 Dirección de recogida:")
        
        imagen = st.file_uploader("📸 Imagen del producto (opcional):", type=["jpg", "jpeg", "png"])
        
        kg_total = KG_EQUIVALENTES[presentacion] * cantidad
        st.info(f"⚖️ Total: **{kg_total} kg**")
        
        submitted = st.form_submit_button("✅ Registrar Producto")
        
        if submitted:
            if not direccion:
                st.error("❌ Por favor ingresa una dirección.")
            else:
                img_nombre = None
                if imagen:
                    img_nombre = guardar_imagen_subida(imagen, prefix=producto)
                
                lat, lon = obtener_coordenadas(ciudad, direccion)
                
                precio_predicho = None
                if IA_CARGADA and modelo_precio:
                    try:
                        entrada = pd.DataFrame({
                            'producto': [producto],
                            'presentacion': [presentacion],
                            'ciudad': [ciudad],
                            'categoria': ['General'],
                            'unidades (kg)': [KG_EQUIVALENTES[presentacion]]
                        })
                        pred = modelo_precio.predict(entrada)
                        try:
                            precio_min, precio_max = sorted(pred[0])
                        except:
                            precio_min = precio_max = float(pred[0])
                        precio_predicho = round((precio_min + precio_max) / 2 * cantidad)
                    except:
                        pass
                
                nueva_notificacion = {
                    'id_notificacion': f"CAMP-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    'fecha_notificacion': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'campesino': campesino,
                    'producto': producto,
                    'cantidad_kg': kg_total,
                    'ciudad': ciudad,
                    'direccion': direccion,
                    'precio': precio_predicho,
                    'precio_predicho': precio_predicho,
                    'calidad': "Buena",
                    'estado': 'Pendiente',
                    'fecha_recogida': None,
                    'transportista_asignado': None,
                    'imagen': img_nombre,
                    'latitud': lat,
                    'longitud': lon,
                    'transportista_lat': None,
                    'transportista_lon': None,
                    'distancia_restante_km': None,
                    'progreso_viaje': 0.0,
                    'tiempo_estimado_llegada': None,
                    'ruta_optimizada': None,
                    'orden_parada': None
                }
                
                df_notif = pd.concat([df_notif, pd.DataFrame([nueva_notificacion])], ignore_index=True)
                guardar_notificaciones(df_notif)
                
                st.success("✅ ¡Producto registrado exitosamente!")
                if precio_predicho:
                    st.info(f"💰 Precio estimado: ${precio_predicho:,.0f} COP")


# VISTA: MIS NOTIFICACIONES CON SEGUIMIENTO EN TIEMPO REAL

def vista_notificaciones():
    st.markdown('<div class="card-venta"><h2>📬 Mis Notificaciones</h2><p>Visualiza el estado de tus productos registrados con seguimiento en tiempo real</p></div>', unsafe_allow_html=True)
    
    df_notif = cargar_notificaciones()
    
    if df_notif.empty:
        st.info("📭 No tienes productos registrados aún.")
        return
    
    # Contar estados
    en_camino = len(df_notif[df_notif['estado'] == 'Aceptado'])
    recogidos = len(df_notif[df_notif['estado'] == 'Recogido'])
    pendientes = len(df_notif[df_notif['estado'] == 'Pendiente'])
    
    # Banner de estado
    if en_camino > 0:
        st.markdown(f"""
        <div class="tracking-live">
            <h3 style='margin: 0; text-align: center;'>
                🚛 {en_camino} transportista(s) en camino | 
                ✅ {recogidos} producto(s) recogido(s) |
                ⏳ {pendientes} pendiente(s)
            </h3>
            <p style='margin: 10px 0 0 0; text-align: center; font-size: 14px;'>
                🔄 Actualización automática activa - Los mapas se refrescan cada {AUTO_REFRESH_INTERVAL} segundos
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Filtros
    col_filtro, col_refresh = st.columns([3, 1])
    
    with col_filtro:
        filtro_estado = st.selectbox(
            "🔍 Filtrar por estado:", 
            ["Todos", "Pendiente", "Aceptado", "Recogido", "Entregado"],
            key="filtro_estado_notif"
        )
    
    with col_refresh:
        if st.button("🔄 Actualizar todo", key="refresh_all_notif", use_container_width=True):
            st.rerun()
    
    # Filtrar datos
    if filtro_estado != "Todos":
        df_filtrado = df_notif[df_notif['estado'] == filtro_estado]
    else:
        df_filtrado = df_notif
    
    st.write(f"**Total de registros:** {len(df_filtrado)}")
    
    # Mostrar cada notificación
    for idx, row in df_filtrado.iterrows():
        # Determinar si expandir automáticamente
        expandir = (row['estado'] == 'Aceptado')
        
        # Color del badge según estado
        badge_color = {
            'Pendiente': '#FFA500',
            'Aceptado': '#EA4335',
            'Recogido': '#34A853',
            'Entregado': '#4285F4'
        }.get(row['estado'], '#666')
        
        titulo_expander = f"🧾 {row['id_notificacion']} - {row['producto']}"
        
        with st.expander(titulo_expander, expanded=expandir):
            # Badge de estado
            st.markdown(f"""
                <span class='status-badge' style='background: {badge_color}; color: white;'>
                    {row['estado'].upper()}
                </span>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**👨‍🌾 Campesino:** {row['campesino']}")
                st.write(f"**📦 Producto:** {row['producto']}")
                st.write(f"**⚖️ Cantidad:** {row['cantidad_kg']} kg")
                st.write(f"**🏙️ Ciudad:** {row['ciudad']}")
                st.write(f"**📍 Dirección:** {row['direccion']}")
                st.write(f"**📅 Fecha:** {row['fecha_notificacion']}")
                
                if pd.notna(row['precio']):
                    st.write(f"**💰 Precio:** ${row['precio']:,.0f} COP")
                
                if pd.notna(row['transportista_asignado']):
                    st.write(f"**🚛 Transportista:** {row['transportista_asignado']}")
                
                # Mostrar progreso si está en camino
                if row['estado'] == 'Aceptado' and pd.notna(row['progreso_viaje']):
                    progreso_valor = float(row['progreso_viaje'])
                    st.markdown("---")
                    st.markdown("### 📊 Progreso del Viaje")
                    st.progress(progreso_valor)
                    st.write(f"**Completado:** {progreso_valor*100:.1f}%")
                    
                    if pd.notna(row['distancia_restante_km']) and pd.notna(row['tiempo_estimado_llegada']):
                        col_dist, col_time = st.columns(2)
                        with col_dist:
                            st.metric("🛣️ Distancia restante", f"{row['distancia_restante_km']:.1f} km")
                        with col_time:
                            st.metric("⏱️ Tiempo estimado", f"~{int(row['tiempo_estimado_llegada'])} min")
                    
                    # Alertas de proximidad
                    if progreso_valor >= 0.95:
                        st.success("🎯 ¡El transportista está muy cerca! Prepara tu producto.")
                    elif progreso_valor >= 0.7:
                        st.warning("📍 El transportista se está acercando.")
                    elif progreso_valor >= 0.3:
                        st.info("🚚 El transportista está en camino.")
            
            with col2:
                # Mostrar imagen del producto
                if pd.notna(row['imagen']):
                    img_path = os.path.join(IMG_DIR, row['imagen'])
                    if os.path.exists(img_path):
                        st.image(img_path, width=200, caption=row['producto'])
            
            # MAPA DE SEGUIMIENTO EN TIEMPO REAL
            if row['estado'] in ['Aceptado', 'Recogido'] and pd.notna(row['transportista_asignado']):
                if validar_coordenadas(row['latitud'], row['longitud']):
                    st.markdown("---")
                    
                    # Mensaje especial si ya fue recogido
                    if row['estado'] == 'Recogido':
                        st.success("✅ **PEDIDO RECOGIDO EXITOSAMENTE** - El transportista tiene tu producto")
                    
                    st.markdown("### 🗺️ Seguimiento en Tiempo Real")
                    
                    # Obtener datos del transportista
                    lat_trans = row.get('transportista_lat')
                    lon_trans = row.get('transportista_lon')
                    progreso = float(row.get('progreso_viaje', 0.0)) if pd.notna(row.get('progreso_viaje')) else 0.0
                    distancia = row.get('distancia_restante_km')
                    
                    # Crear mapa interactivo
                    if validar_coordenadas(lat_trans, lon_trans):
                        mapa = crear_mapa_seguimiento_folium(
                            lat_campesino=row['latitud'],
                            lon_campesino=row['longitud'],
                            lat_transportista=lat_trans,
                            lon_transportista=lon_trans,
                            distancia_km=distancia if pd.notna(distancia) else None,
                            progreso=progreso,
                            transportista_nombre=row['transportista_asignado']
                        )
                        
                        # Mostrar el mapa
                        st_folium(mapa, width=900, height=550, key=f"mapa_{row['id_notificacion']}")
                        
                        # Leyenda del mapa
                        st.markdown("""
                            <div style='text-align: center; margin-top: 15px; padding: 15px; background: white; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);'>
                                <span class='legend-item' style='color: #34A853;'>🟢 Tu ubicación</span>
                                <span class='legend-item' style='color: #EA4335;'>🔴 Transportista</span>
                                <span class='legend-item' style='color: #4285F4;'>🔵 Ruta restante</span>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Botón de actualización manual
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.button(f"🔄 Actualizar ubicación del transportista", 
                                    key=f"refresh_{row['id_notificacion']}", 
                                    use_container_width=True):
                            st.rerun()
                    else:
                        st.info("🚛 El transportista aceptó tu pedido. En breve verás su ubicación en el mapa.")
                else:
                    st.warning("⚠️ No se pudo obtener la geolocalización de tu dirección.")


# VISTA: VENTA RÁPIDA IA
def vista_venta_rapida():
    st.markdown('<div class="card-venta"><h2>🤖 Venta Rápida con IA</h2><p>Sube una foto y deja que la inteligencia artificial identifique tu producto y estime su precio</p></div>', unsafe_allow_html=True)

    df_notif = cargar_notificaciones()

    imagen = st.file_uploader(
        "📸 Sube una imagen del producto",
        type=["jpg", "jpeg", "png"],
        key="venta_rapida_uploader"
    )

    if imagen is None and st.session_state.get("venta_producto"):
        st.session_state.venta_producto = None
        st.session_state.venta_confianza = None
        st.session_state.venta_img_nombre = None
        st.session_state.venta_resultados = []
        st.rerun()

    if 'venta_producto' not in st.session_state:
        st.session_state.venta_producto = None
        st.session_state.venta_confianza = None
        st.session_state.venta_img_nombre = None
        st.session_state.venta_resultados = []

    producto_detectado = st.session_state.venta_producto
    confianza = st.session_state.venta_confianza
    img_nombre = st.session_state.venta_img_nombre
    resultados_presentaciones = st.session_state.venta_resultados

    if imagen and producto_detectado is None:
        if modelo_clasificacion is None:
            st.error("❌ El modelo de clasificación no está disponible.")
            return

        img_nombre = guardar_imagen_subida(imagen, prefix="venta_rapida")
        st.session_state.venta_img_nombre = img_nombre

        st.info("🧠 Analizando la imagen con IA real...")

        img_path = os.path.join(IMG_DIR, img_nombre)
        producto_detectado, confianza = predecir_producto_real(img_path)

        if producto_detectado == "Error":
            return

        st.session_state.venta_producto = producto_detectado
        st.session_state.venta_confianza = confianza

        presentaciones_validas = PRESENTACIONES.get(producto_detectado, ["Kilogramo (1 kg)"])
        
        if producto_detectado not in PRESENTACIONES:
            st.warning(f"⚠️ El producto '{producto_detectado}' no está en el catálogo. Usando presentación genérica.")
        
        resultados_presentaciones = []

        if IA_CARGADA and modelo_precio is not None:
            for nombre_pres in presentaciones_validas:
                kg = KG_EQUIVALENTES.get(nombre_pres, 1)
                try:
                    entrada = pd.DataFrame({
                        'producto': [producto_detectado],
                        'presentacion': [nombre_pres],
                        'ciudad': ['Tunja'],
                        'categoria': ['General'],
                        'unidades (kg)': [kg]
                    })

                    pred = modelo_precio.predict(entrada)
                    resultado = pred[0]

                    try:
                        precio_min_total, precio_max_total = sorted(resultado)
                    except:
                        precio_min_total = precio_max_total = float(resultado)

                    precio_min_kg = precio_min_total / kg
                    precio_max_kg = precio_max_total / kg
                    precio_promedio_kg = (precio_min_kg + precio_max_kg) / 2

                    resultados_presentaciones.append({
                        "presentacion": nombre_pres,
                        "kg": kg,
                        "precio_min": float(precio_min_kg),
                        "precio_max": float(precio_max_kg),
                        "precio_promedio": float(precio_promedio_kg)
                    })
                except Exception as e:
                    st.warning(f"⚠️ Error al calcular precio para {nombre_pres}: {e}")

        st.session_state.venta_resultados = resultados_presentaciones

    if producto_detectado and img_nombre:
        img_path = os.path.join(IMG_DIR, img_nombre)
        if os.path.exists(img_path):
            st.image(img_path, width=300, caption=f"📷 {producto_detectado}")

        st.success(f"🎯 Producto detectado: **{producto_detectado}** ({confianza*100:.1f}% confianza)")
        
        with st.expander("🔍 Información de Debug"):
            st.markdown(f"""
            <div class="debug-box">
                <b>📦 Producto detectado:</b> {producto_detectado}<br>
                <b>✅ Está en PRESENTACIONES:</b> {'Sí' if producto_detectado in PRESENTACIONES else 'No'}<br>
                <b>📋 Presentaciones disponibles:</b> {', '.join(PRESENTACIONES.get(producto_detectado, ['Ninguna']))}<br>
                <b>🔢 Número de opciones de precio:</b> {len(resultados_presentaciones)}
            </div>
            """, unsafe_allow_html=True)

    if not imagen and producto_detectado is None:
        st.info("📷 Sube una imagen para que la IA intente detectar el producto.")
        return

    if resultados_presentaciones:
        st.markdown("### 💰 Estimaciones de precios")

        col1, col2 = st.columns(2)
        for i, res in enumerate(resultados_presentaciones):
            with (col1 if i % 2 == 0 else col2):
                precio_presentacion = res['precio_promedio'] * res['kg']
                rango_min = res['precio_min'] * res['kg']
                rango_max = res['precio_max'] * res['kg']
                
                st.markdown(f"""
                    <div class="precio-card">
                        <h4>📦 {res['presentacion']}</h4>
                        <p style="font-size:24px; color:#667eea; font-weight:bold;">💵 ${precio_presentacion:,.0f} COP</p>
                        <p style="font-size:14px; color:#333;"><b>Precio por unidad de presentación</b></p>
                        <p>📊 Rango: ${rango_min:,.0f} - ${rango_max:,.0f}</p>
                        <p>⚖️ Peso: {res['kg']} kg por unidad</p>
                        <p style="font-size:12px; color:#999;">💡 Equivale a ${res['precio_promedio']:,.0f} COP/kg</p>
                    </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🟢 Registrar producto detectado")

        with st.form("form_venta_rapida_ia"):
            st.markdown("#### 📦 Información del producto")
            col_prod1, col_prod2 = st.columns(2)
            
            with col_prod1:
                presentaciones_validas = PRESENTACIONES.get(producto_detectado, ["Kilogramo (1 kg)"])
                
                presentacion_seleccionada = st.selectbox(
                    "📦 Selecciona la presentación:",
                    options=presentaciones_validas,
                    key="venta_presentacion_select"
                )

                kg_seleccionado = KG_EQUIVALENTES.get(presentacion_seleccionada, 1)
                precio_por_kg = next(
                    (r["precio_promedio"] for r in resultados_presentaciones if r["presentacion"] == presentacion_seleccionada),
                    0
                )
                precio_unitario = precio_por_kg * kg_seleccionado
            
            with col_prod2:
                cantidad = st.number_input(
                    "🔢 Cantidad de presentaciones a vender",
                    min_value=1,
                    max_value=100,
                    value=1,
                    step=1,
                    key="venta_cantidad"
                )
            
            kg_total = kg_seleccionado * cantidad
            precio_total = precio_unitario * cantidad

            st.markdown("---")
            st.markdown("#### 💰 Resumen de compra")
            
            col_precio1, col_precio2, col_precio3 = st.columns(3)
            with col_precio1:
                st.metric(
                    label="💰 Precio unitario",
                    value=f"${precio_unitario:,.0f}",
                    help=f"Precio por cada {presentacion_seleccionada}"
                )
            with col_precio2:
                st.metric(
                    label="⚖️ Peso total",
                    value=f"{kg_total} kg",
                    help=f"{cantidad} presentación(es) × {kg_seleccionado} kg"
                )
            with col_precio3:
                st.metric(
                    label="💵 TOTAL a pagar",
                    value=f"${precio_total:,.0f}",
                    help=f"{cantidad} × ${precio_unitario:,.0f}"
                )
            
            if cantidad > 1:
                st.info(f"📦 Estás vendiendo **{cantidad} {presentacion_seleccionada}** por un total de **${precio_total:,.0f} COP**")
            
            st.markdown("---")
            
            st.markdown("#### 📍 Información de ubicación y campesino")
            col_ubi1, col_ubi2 = st.columns(2)
            
            with col_ubi1:
                campesino_nombre = st.text_input(
                    "👨‍🌾 Nombre del campesino:",
                    value="Campesino",
                    key="venta_campesino"
                )
                ciudad_venta = st.text_input(
                    "🏙️ Ciudad:",
                    value="Tunja",
                    key="venta_ciudad"
                )
            
            with col_ubi2:
                direccion_venta = st.text_area(
                    "📍 Dirección de recogida:",
                    placeholder="Ej: Carrera 10 #15-25, Barrio Centro",
                    key="venta_direccion",
                    height=100
                )
            
            st.info("ℹ️ La dirección será geocodificada automáticamente para el seguimiento del transportista.")

            submitted_venta = st.form_submit_button("✅ Registrar venta con ubicación", use_container_width=True, type="primary")
            
            if submitted_venta:
                if not direccion_venta.strip():
                    st.error("❌ Por favor ingresa una dirección válida.")
                elif not campesino_nombre.strip():
                    st.error("❌ Por favor ingresa el nombre del campesino.")
                else:
                    with st.spinner("🌍 Obteniendo coordenadas de la dirección..."):
                        lat, lon = obtener_coordenadas(ciudad_venta, direccion_venta)
                    
                    if lat is None or lon is None:
                        st.warning("⚠️ No se pudo obtener la geolocalización automáticamente.")
                    else:
                        st.success(f"✅ Ubicación encontrada: {lat:.6f}, {lon:.6f}")
                    
                    nueva_notificacion = {
                        'id_notificacion': f"AI-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        'fecha_notificacion': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'campesino': campesino_nombre.strip(),
                        'producto': producto_detectado,
                        'cantidad_kg': kg_total,
                        'ciudad': ciudad_venta.strip(),
                        'direccion': direccion_venta.strip(),
                        'precio': round(precio_total),
                        'precio_predicho': round(precio_total),
                        'calidad': "Buena",
                        'estado': 'Pendiente',
                        'fecha_recogida': None,
                        'transportista_asignado': None,
                        'imagen': img_nombre,
                        'latitud': lat,
                        'longitud': lon,
                        'transportista_lat': None,
                        'transportista_lon': None,
                        'distancia_restante_km': None,
                        'progreso_viaje': 0.0,
                        'tiempo_estimado_llegada': None,
                        'ruta_optimizada': None,
                        'orden_parada': None
                    }

                    df_notif = pd.concat([df_notif, pd.DataFrame([nueva_notificacion])], ignore_index=True)
                    guardar_notificaciones(df_notif)

                    st.success("✅ ¡Venta registrada exitosamente!")
                    st.balloons()

                    st.session_state.venta_producto = None
                    st.session_state.venta_confianza = None
                    st.session_state.venta_img_nombre = None
                    st.session_state.venta_resultados = []

                    time.sleep(2)
                    st.rerun()
                    
                    
# FUNCIÓN PRINCIPAL

def view_campesino():
    aplicar_estilos()
    inicializar_session_state()
    
    st.sidebar.title("🌾 Menú Campesino")
    st.sidebar.write("**Estado del sistema:**")
    if IA_CARGADA and modelo_precio:
        st.sidebar.success("✅ IA Activa")
    else:
        st.sidebar.warning("⚠️ IA Limitada")
    
    df_notif = cargar_notificaciones()
    en_camino = len(df_notif[df_notif['estado'] == 'Aceptado'])
    pendientes = len(df_notif[df_notif['estado'] == 'Pendiente'])
    recogidos = len(df_notif[df_notif['estado'] == 'Recogido'])
    entregados = len(df_notif[df_notif['estado'] == 'Entregado'])
    
    st.sidebar.markdown("---")
    st.sidebar.write("**📊 Resumen de pedidos:**")
    st.sidebar.metric("⏳ Pendientes", pendientes)
    st.sidebar.metric("🚛 En camino", en_camino)
    st.sidebar.metric("✅ Recogidos", recogidos)
    st.sidebar.metric("📦 Entregados", entregados)
    
    # Control de auto-refresh
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔄 Actualización Automática")
    
    auto_refresh = st.sidebar.checkbox(
        "Activar auto-actualización", 
        value=st.session_state.auto_refresh_enabled,
        key="auto_refresh_campesino",
        help=f"Actualiza automáticamente cada {AUTO_REFRESH_INTERVAL} segundos"
    )
    st.session_state.auto_refresh_enabled = auto_refresh
    
    if auto_refresh:
        st.sidebar.success(f"✅ Activo ({AUTO_REFRESH_INTERVAL}s)")
        if en_camino > 0:
            st.sidebar.info(f"🚛 {en_camino} transportista(s) en seguimiento")
        
        # JavaScript para auto-refresh
        st.markdown(f"""
            <script>
                setTimeout(function(){{
                    window.parent.location.reload();
                }}, {AUTO_REFRESH_INTERVAL * 1000});
            </script>
        """, unsafe_allow_html=True)
        
        st.session_state.refresh_counter += 1
        st.sidebar.caption(f"Actualización #{st.session_state.refresh_counter}")
    else:
        st.sidebar.warning("⏸️ Pausado")
    
    if st.sidebar.button("🔃 Actualizar Ahora", use_container_width=True):
        st.rerun()
    
    # Tabs principales
    tab1, tab2, tab3 = st.tabs([
        "📝 Registro de Productos",
        "📬 Mis Notificaciones",
        "🤖 Venta Rápida con IA"
    ])
    
    with tab1:
        vista_registro()
    
    with tab2:
        vista_notificaciones()
    
    with tab3:
        vista_venta_rapida()


# EJECUCIÓN DIRECTA

if __name__ == "__main__":
    st.set_page_config(
        page_title="Sistema Campesino - Seguimiento en Tiempo Real",
        page_icon="🌾",
        layout="wide"
    )
    view_campesino()