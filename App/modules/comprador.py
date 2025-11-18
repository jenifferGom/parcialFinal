import streamlit as stA
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from collections import Counter
import plotly.graph_objects as go
import plotly.express as px


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN DE RUTAS 
# ═══════════════════════════════════════════════════════════════════════════

APP_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.join(APP_ROOT, "data")
IMG_DIR = os.path.join(APP_ROOT, "modules", "imagenes_productos")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

CSV_NOTIFICACIONES = os.path.join(DATA_DIR, "notificaciones_transporte.csv")
CSV_COMPRAS = os.path.join(DATA_DIR, "historial_compras.csv")
CSV_ALERTAS = os.path.join(DATA_DIR, "alertas_precios.csv")


# ═══════════════════════════════════════════════════════════════════════════
#  FUNCIONES DE CARGA/GUARDADO 
# ═══════════════════════════════════════════════════════════════════════════

def cargar_notificaciones():
    """Carga las notificaciones/productos desde el CSV"""
    columnas_base = [
        'producto', 'ciudad', 'cantidad_kg', 'estado', 'calidad',
        'campesino', 'transportador', 'transportista_asignado', 'precio_predicho', 
        'origen', 'imagen', 'fecha_notificacion', 'precio'
    ]
    
    if not os.path.exists(CSV_NOTIFICACIONES):
        return pd.DataFrame(columns=columnas_base)
    
    try:
        df = pd.read_csv(CSV_NOTIFICACIONES)
    except Exception as e:
        st.error(f"❌ Error al leer notificaciones: {e}")
        return pd.DataFrame(columns=columnas_base)

    if df.empty:
        return pd.DataFrame(columns=columnas_base)

    for col in columnas_base:
        if col not in df.columns:
            df[col] = None

    # Asignar origen basado en transportista_asignado o transportador
    # Si un transportista recogió el producto (estado='Recogido'), es su producto para vender
    if 'transportista_asignado' in df.columns:
        # Productos recogidos por transportista son vendidos por él
        df.loc[
            (df['estado'] == 'Recogido') & 
            (df['transportista_asignado'].notna()) & 
            (df['transportista_asignado'] != ''), 
            'origen'
        ] = 'Transportador'
        
        # Guardar el nombre del transportista como vendedor
        df.loc[
            (df['estado'] == 'Recogido') & 
            (df['transportista_asignado'].notna()) & 
            (df['transportista_asignado'] != ''), 
            'transportador'
        ] = df['transportista_asignado']
    
    # Para productos que ya tienen transportador asignado
    if 'transportador' in df.columns:
        df.loc[df['transportador'].notna() & (df['transportador'] != ''), 'origen'] = 'Transportador'
    
    # Por defecto, los productos pendientes son del campesino
    df.loc[df['origen'].isna() | (df['origen'] == ''), 'origen'] = 'Campesino'

    return df


def guardar_notificaciones(df):
    """Guarda el DataFrame de notificaciones"""
    try:
        df.to_csv(CSV_NOTIFICACIONES, index=False)
    except Exception as e:
        st.error(f"❌ Error al guardar notificaciones: {e}")


def cargar_historial_compras():
    """Carga el historial de compras"""
    if not os.path.exists(CSV_COMPRAS):
        return pd.DataFrame(columns=[
            'fecha_compra', 'comprador', 'vendedor', 'origen',
            'producto', 'cantidad_kg', 'precio_unitario', 'precio_total',
            'ciudad', 'calificacion', 'comentario'
        ])
    try:
        df = pd.read_csv(CSV_COMPRAS)
        if 'fecha_compra' in df.columns:
            df['fecha_compra'] = pd.to_datetime(df['fecha_compra'])
        return df
    except:
        return pd.DataFrame(columns=[
            'fecha_compra', 'comprador', 'vendedor', 'origen',
            'producto', 'cantidad_kg', 'precio_unitario', 'precio_total',
            'ciudad', 'calificacion', 'comentario'
        ])


def cargar_alertas():
    """Carga alertas de precio configuradas"""
    if not os.path.exists(CSV_ALERTAS):
        return pd.DataFrame(columns=[
            'comprador', 'producto', 'ciudad', 'precio_objetivo', 'activa', 'fecha_creacion'
        ])
    try:
        return pd.read_csv(CSV_ALERTAS)
    except:
        return pd.DataFrame(columns=[
            'comprador', 'producto', 'ciudad', 'precio_objetivo', 'activa', 'fecha_creacion'
        ])


def guardar_alertas(df):
    """Guarda alertas de precio"""
    try:
        df.to_csv(CSV_ALERTAS, index=False)
    except Exception as e:
        st.error(f"❌ Error al guardar alertas: {e}")


# ═══════════════════════════════════════════════════════════════════════════
#  FUNCIONES DE REPUTACIÓN 
# ═══════════════════════════════════════════════════════════════════════════

def calcular_reputacion(vendedor_nombre, df_productos):
    """Calcula el score de reputación general"""
    if df_productos.empty or vendedor_nombre is None:
        return {
            'score': 0, 
            'total_productos': 0, 
            'calidad_buena_pct': 0, 
            'precio_promedio': 0, 
            'productos_entregados': 0, 
            'tasa_entrega': 0
        }

    productos_vendedor = df_productos[
        (df_productos['campesino'] == vendedor_nombre) |
        (df_productos.get('transportador', pd.Series(dtype='object')) == vendedor_nombre)
    ]

    if productos_vendedor.empty:
        return {
            'score': 0, 
            'total_productos': 0, 
            'calidad_buena_pct': 0, 
            'precio_promedio': 0, 
            'productos_entregados': 0, 
            'tasa_entrega': 0
        }

    total = len(productos_vendedor)
    buena_calidad = len(productos_vendedor[productos_vendedor['calidad'] == 'Buena'])
    entregados = len(productos_vendedor[productos_vendedor['estado'].isin(['Completado', 'Recogido', 'Entregado'])])

    calidad_pct = (buena_calidad / total * 100) if total > 0 else 0
    entrega_pct = (entregados / total * 100) if total > 0 else 0
    precio_promedio = productos_vendedor['precio_predicho'].mean() if 'precio_predicho' in productos_vendedor.columns else 0

    score = (calidad_pct * 0.5 + entrega_pct * 0.3 + min(total / 10 * 20, 20))

    return {
        'score': round(score, 1),
        'total_productos': total,
        'calidad_buena_pct': round(calidad_pct, 1),
        'precio_promedio': round(precio_promedio, 2) if not pd.isna(precio_promedio) else 0,
        'productos_entregados': entregados,
        'tasa_entrega': round(entrega_pct, 1)
    }


def recomendar_proveedores_por_calidad(producto, ciudad, df_productos, top_n=5):
    """Recomienda proveedores priorizando la calidad histórica"""
    if df_productos.empty:
        return []

    df_filtrado = df_productos[
        (df_productos['producto'].str.lower() == producto.lower()) &
        (df_productos['ciudad'].str.lower() == ciudad.lower())
    ].copy()

    if df_filtrado.empty:
        return []

    vendedores = set()
    for _, row in df_filtrado.iterrows():
        if row['origen'] == "Campesino" and pd.notna(row['campesino']):
            vendedores.add(row['campesino'])
        elif row['origen'] == "Transportador" and pd.notna(row.get('transportador')):
            vendedores.add(row['transportador'])

    recomendaciones = []
    for vendedor in vendedores:
        reputacion = calcular_reputacion(vendedor, df_productos)
        productos_vend = df_filtrado[
            (df_filtrado['campesino'] == vendedor) |
            (df_filtrado.get('transportador', pd.Series(dtype='object')) == vendedor)
        ]
        calidad_promedio = productos_vend['calidad'].apply(lambda x: 1 if x == 'Buena' else 0).mean() * 100

        tipo_vendedor = "Campesino" if productos_vend.iloc[0]['origen'] == "Campesino" else "Transportador"

        recomendaciones.append({
            'vendedor': vendedor,
            'tipo': tipo_vendedor,
            'score': reputacion['score'],
            'calidad_historica_pct': round(calidad_promedio, 1),
            'total_productos': reputacion['total_productos'],
            'precio_promedio': round(productos_vend['precio_predicho'].mean(), 2) if not productos_vend['precio_predicho'].isna().all() else 0,
            'productos_disponibles': len(productos_vend[productos_vend['estado'] == 'Pendiente'])
        })

    recomendaciones = sorted(recomendaciones, key=lambda x: (x['calidad_historica_pct'], x['score']), reverse=True)
    return recomendaciones[:top_n]


# ═══════════════════════════════════════════════════════════════════════════
#  FUNCIONES DE ANÁLISIS 
# ═══════════════════════════════════════════════════════════════════════════

def analizar_tendencia_precios(producto, ciudad, df_productos):
    """Analiza tendencias de precios por mes"""
    if df_productos.empty or 'fecha_notificacion' not in df_productos.columns:
        return None
    
    df_filtrado = df_productos[
        (df_productos['producto'].str.lower() == producto.lower()) &
        (df_productos['ciudad'].str.lower() == ciudad.lower()) &
        (df_productos['precio_predicho'].notna())
    ].copy()
    
    if df_filtrado.empty:
        return None
    
    try:
        df_filtrado['fecha'] = pd.to_datetime(df_filtrado['fecha_notificacion'])
        df_filtrado['mes'] = df_filtrado['fecha'].dt.to_period('M')
        
        tendencia = df_filtrado.groupby('mes').agg({
            'precio_predicho': ['mean', 'min', 'max', 'count']
        }).reset_index()
        
        tendencia.columns = ['mes', 'precio_promedio', 'precio_min', 'precio_max', 'cantidad']
        tendencia['mes'] = tendencia['mes'].astype(str)
        
        return tendencia
    except:
        return None


def sugerir_mejor_momento_compra(tendencia_df):
    """Sugiere el mejor momento para comprar basado en tendencias"""
    if tendencia_df is None or tendencia_df.empty:
        return None
    
    idx_min = tendencia_df['precio_promedio'].idxmin()
    idx_max = tendencia_df['precio_promedio'].idxmax()
    
    mes_barato = tendencia_df.loc[idx_min, 'mes']
    precio_barato = tendencia_df.loc[idx_min, 'precio_promedio']
    
    mes_caro = tendencia_df.loc[idx_max, 'mes']
    precio_caro = tendencia_df.loc[idx_max, 'precio_promedio']
    
    ahorro_potencial = ((precio_caro - precio_barato) / precio_caro) * 100
    
    return {
        'mes_barato': mes_barato,
        'precio_barato': precio_barato,
        'mes_caro': mes_caro,
        'precio_caro': precio_caro,
        'ahorro_pct': round(ahorro_potencial, 1)
    }


def registrar_compra(comprador, producto_data):
    """Registra una compra en el historial"""
    try:
        if not os.path.exists(CSV_COMPRAS):
            df_compras = pd.DataFrame(columns=[
                'fecha_compra', 'comprador', 'vendedor', 'origen',
                'producto', 'cantidad_kg', 'precio_unitario', 'precio_total',
                'ciudad', 'calificacion', 'comentario'
            ])
        else:
            df_compras = pd.read_csv(CSV_COMPRAS)

        vendedor = producto_data.get('campesino') if producto_data['origen'] == "Campesino" else producto_data.get('transportador', 'Transportador')

        nueva_compra = {
            'fecha_compra': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'comprador': comprador,
            'vendedor': vendedor,
            'origen': producto_data['origen'],
            'producto': producto_data['producto'],
            'cantidad_kg': producto_data['cantidad_kg'],
            'precio_unitario': producto_data['precio_unitario'],
            'precio_total': producto_data['cantidad_kg'] * producto_data['precio_unitario'],
            'ciudad': producto_data['ciudad'],
            'calificacion': producto_data.get('calificacion', 0),
            'comentario': producto_data.get('comentario', '')
        }

        df_compras = pd.concat([df_compras, pd.DataFrame([nueva_compra])], ignore_index=True)
        df_compras.to_csv(CSV_COMPRAS, index=False)
        return True
    except Exception as e:
        st.error(f"❌ Error al registrar la compra: {e}")
        return False


def verificar_alertas(comprador, df_productos, df_alertas):
    """Verifica si hay productos que cumplen con alertas de precio"""
    if df_alertas.empty:
        return []
    
    alertas_activas = df_alertas[
        (df_alertas['comprador'] == comprador) & 
        (df_alertas['activa'] == True)
    ]
    
    productos_alerta = []
    
    for _, alerta in alertas_activas.iterrows():
        productos_disponibles = df_productos[
            (df_productos['producto'] == alerta['producto']) &
            (df_productos['ciudad'] == alerta['ciudad']) &
            (df_productos['estado'] == 'Pendiente') &
            (df_productos['precio_predicho'] <= alerta['precio_objetivo'])
        ]
        
        if not productos_disponibles.empty:
            productos_alerta.append({
                'producto': alerta['producto'],
                'ciudad': alerta['ciudad'],
                'precio_objetivo': alerta['precio_objetivo'],
                'cantidad_disponible': len(productos_disponibles),
                'precio_min': productos_disponibles['precio_predicho'].min()
            })
    
    return productos_alerta


# ═══════════════════════════════════════════════════════════════════════════
# ESTILOS CSS 
# ═══════════════════════════════════════════════════════════════════════════

def aplicar_estilos():
    st.markdown("""
        <style>
        .titulo-comprador { 
            font-size: 2.5rem; 
            font-weight: bold; 
            text-align: center; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            padding: 20px; 
        }
        .producto-card { 
            background: white; 
            border: 2px solid #3498db; 
            border-radius: 15px; 
            padding: 20px; 
            margin: 15px 0; 
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .producto-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        }
        .badge-campesino {
            background: linear-gradient(135deg, #27ae60 0%, #229954 100%);
            color: white;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: bold;
            display: inline-block;
            margin: 5px 5px 5px 0;
        }
        .badge-transportador {
            background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
            color: white;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: bold;
            display: inline-block;
            margin: 5px 5px 5px 0;
        }
        .badge-estado {
            background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
            color: white;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: bold;
            display: inline-block;
            margin: 5px 5px 5px 0;
        }
        .mejor-precio { 
            background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%); 
            color: white; 
            padding: 8px 18px; 
            border-radius: 25px; 
            font-weight: bold;
            display: inline-block;
        }
        .info-comprador {
            background: linear-gradient(135deg, #ecf0f1 0%, #d5dbdb 100%);
            padding: 20px;
            border-radius: 12px;
            border-left: 5px solid #3498db;
            margin-bottom: 25px;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin: 10px 0;
        }
        .stat-value {
            font-size: 32px;
            font-weight: bold;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .stat-label {
            font-size: 14px;
            color: #5f6368;
            text-transform: uppercase;
            margin-top: 5px;
        }
        .alerta-precio {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
            color: white;
            padding: 15px;
            border-radius: 12px;
            margin: 10px 0;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.8; }
        }
        </style>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# INTERFAZ PRINCIPAL 
# ═══════════════════════════════════════════════════════════════════════════

def view_comprador():
    aplicar_estilos()
    
    st.markdown('<div class="titulo-comprador">🛒 Marketplace Agrícola</div>', unsafe_allow_html=True)
    
    # Información del sistema
    st.markdown("""
        <div class="info-comprador">
            <h3 style='margin-top:0;'>📦 Plataforma de Compra Directa</h3>
            <p>Conecta directamente con productores y obtén los mejores precios:</p>
            <span class="badge-campesino">🌾 Campesinos</span> - Productos directos del campo<br>
            <span class="badge-transportador">🚛 Transportadores</span> - Productos en tránsito con disponibilidad inmediata
        </div>
    """, unsafe_allow_html=True)

    # Cargar datos
    df_productos = cargar_notificaciones()
    df_compras = cargar_historial_compras()
    df_alertas = cargar_alertas()

    # Verificar si hay productos
    if df_productos.empty:
        st.warning("⚠️ No hay productos disponibles actualmente.")
        return

    # Sidebar - Perfil del comprador
    with st.sidebar:
        st.markdown("### 👤 Perfil del Comprador")
        nombre_comprador = st.text_input("Tu nombre:", value="Comprador", key="nombre_comprador_sidebar")
        
        # Verificar alertas
        alertas_disparadas = verificar_alertas(nombre_comprador, df_productos, df_alertas)
        
        if alertas_disparadas:
            st.markdown("---")
            st.markdown("### 🔔 ¡Alertas de Precio!")
            for alerta in alertas_disparadas:
                st.markdown(f"""
                <div class="alerta-precio">
                    <b>🎯 {alerta['producto']} en {alerta['ciudad']}</b><br>
                    💰 Desde ${alerta['precio_min']:.0f}/kg<br>
                    📦 {alerta['cantidad_disponible']} disponibles
                </div>
                """, unsafe_allow_html=True)
        
        # Estadísticas del comprador
        if not df_compras.empty:
            compras_usuario = df_compras[df_compras['comprador'] == nombre_comprador]
            if not compras_usuario.empty:
                st.markdown("---")
                st.markdown("### 📊 Mis Estadísticas")
                st.metric("🛍️ Total Compras", len(compras_usuario))
                st.metric("💰 Gasto Total", f"${compras_usuario['precio_total'].sum():,.0f}")
                
                if 'calificacion' in compras_usuario.columns:
                    promedio_calif = compras_usuario[compras_usuario['calificacion'] > 0]['calificacion'].mean()
                    if not pd.isna(promedio_calif):
                        st.metric("⭐ Calificación Promedio", f"{promedio_calif:.1f}/5")

    # Tabs principales
    tabs = st.tabs([
        "🔍 Buscar Productos", 
        "⭐ Proveedores Recomendados", 
        "🛍️ Realizar Compra",
        "📈 Análisis de Precios",
        "🔔 Mis Alertas",
        "📜 Mi Historial"
    ])

    # ═══════════════════════════════════════════════════════════════════════
    #  TAB 1: BUSCAR PRODUCTOS 
    # ═══════════════════════════════════════════════════════════════════════
    with tabs[0]:
        st.subheader("🔍 Catálogo de Productos Disponibles")
        
        col1, col2, col3 = st.columns(3)
        productos = sorted(df_productos['producto'].dropna().unique().tolist())
        ciudades = sorted(df_productos['ciudad'].dropna().unique().tolist())
        
        producto_sel = col1.selectbox("Producto:", ["Todos"] + productos, key="buscar_producto")
        ciudad_sel = col2.selectbox("Ciudad:", ["Todas"] + ciudades, key="buscar_ciudad")
        origen_sel = col3.selectbox("Tipo:", ["Todos", "Campesino", "Transportador"], key="buscar_origen")

        df_filtro = df_productos[df_productos['estado'] == 'Pendiente']
        
        # IMPORTANTE: También incluir productos "Recogidos" por transportistas (disponibles para venta)
        df_filtro_recogidos = df_productos[
            (df_productos['estado'] == 'Recogido') & 
            (df_productos['origen'] == 'Transportador')
        ]
        
        # Combinar ambos filtros
        df_filtro = pd.concat([df_filtro, df_filtro_recogidos], ignore_index=True)
        
        if producto_sel != "Todos":
            df_filtro = df_filtro[df_filtro['producto'] == producto_sel]
        if ciudad_sel != "Todas":
            df_filtro = df_filtro[df_filtro['ciudad'] == ciudad_sel]
        if origen_sel != "Todos":
            df_filtro = df_filtro[df_filtro['origen'] == origen_sel]

        if df_filtro.empty:
            st.info("📭 No hay productos que coincidan con los filtros.")
        else:
            # Ordenar por precio
            orden = st.radio("Ordenar por:", ["Precio: Menor a Mayor", "Precio: Mayor a Menor", "Más Recientes"], horizontal=True)
            
            if orden == "Precio: Menor a Mayor":
                df_filtro = df_filtro.sort_values('precio_predicho', ascending=True)
            elif orden == "Precio: Mayor a Menor":
                df_filtro = df_filtro.sort_values('precio_predicho', ascending=False)
            else:
                df_filtro = df_filtro.sort_values('fecha_notificacion', ascending=False)
            
            st.success(f"✅ Se encontraron **{len(df_filtro)}** productos disponibles")
            
            # Mostrar productos
            for idx, row in df_filtro.iterrows():
                st.markdown('<div class="producto-card">', unsafe_allow_html=True)
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    ruta = os.path.join(IMG_DIR, str(row.get('imagen', '')))
                    if pd.notna(row.get('imagen')) and os.path.exists(ruta):
                        st.image(ruta, use_container_width=True)
                    else:
                        st.info("📷 Sin imagen")
                
                with col2:
                    st.markdown(f"### {row['producto']}")
                    
                    # Badge de tipo de vendedor
                    if row['origen'] == "Campesino":
                        st.markdown('<span class="badge-campesino">🌾 Campesino - Producto en Finca</span>', unsafe_allow_html=True)
                        vendedor = row['campesino']
                    else:
                        st.markdown('<span class="badge-transportador">🚛 Transportador - Producto Disponible</span>', unsafe_allow_html=True)
                        vendedor = row.get('transportador', row.get('transportista_asignado', 'Transportador'))
                    
                    # FIX: Crear estado_badge basado en el estado del producto
                    estado_actual = row['estado']
                    if estado_actual == 'Pendiente':
                        estado_badge = '<span class="badge-estado">📍 En Finca</span>'
                    elif estado_actual == 'Recogido':
                        estado_badge = '<span class="badge-estado">🚛 Listo para Entrega</span>'
                    else:
                        estado_badge = f'<span class="badge-estado">{estado_actual}</span>'
                    
                    col_info1, col_info2 = st.columns(2)
                    
                    with col_info1:
                        st.write(f"**👤 Vendedor:** {vendedor}")
                        st.write(f"**📍 Ciudad:** {row['ciudad']}")
                        st.write(f"**⚖️ Disponible:** {row['cantidad_kg']} kg")
                        st.markdown(f"{estado_badge}", unsafe_allow_html=True)
                    
                    with col_info2:
                        st.write(f"**💰 Precio:** ${row['precio_predicho']:.2f}/kg")
                        st.write(f"**🏆 Calidad:** {row['calidad']}")
                        
                        # Calcular reputación
                        rep = calcular_reputacion(vendedor, df_productos)
                        st.write(f"**⭐ Reputación:** {rep['score']:.0f}/100")
                
                st.markdown('</div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════
    #  TAB 2: PROVEEDORES RECOMENDADOS 
    # ═══════════════════════════════════════════════════════════════════════
    with tabs[1]:
        st.subheader("⭐ Ranking de Proveedores por Calidad")
        
        productos_list = sorted(df_productos['producto'].dropna().unique().tolist())
        ciudades_list = sorted(df_productos['ciudad'].dropna().unique().tolist())
        
        if not productos_list or not ciudades_list:
            st.info("📭 No hay suficientes datos para análisis.")
        else:
            col1, col2 = st.columns(2)
            producto_rec = col1.selectbox("Producto:", productos_list, key="rec_producto")
            ciudad_rec = col2.selectbox("Ciudad:", ciudades_list, key="rec_ciudad")

            if st.button("🔎 Analizar Proveedores", type="primary", use_container_width=True):
                recomendaciones = recomendar_proveedores_por_calidad(producto_rec, ciudad_rec, df_productos)
                
                if not recomendaciones:
                    st.info("📭 No se encontraron proveedores.")
                else:
                    st.success(f"✅ **{len(recomendaciones)}** proveedores encontrados")
                    
                    # Gráfico de calidad
                    fig = go.Figure()
                    
                    campesinos = [r for r in recomendaciones if r['tipo'] == 'Campesino']
                    transportadores = [r for r in recomendaciones if r['tipo'] == 'Transportador']
                    
                    if campesinos:
                        fig.add_trace(go.Bar(
                            name='🌾 Campesinos',
                            x=[r['vendedor'] for r in campesinos],
                            y=[r['calidad_historica_pct'] for r in campesinos],
                            marker_color='#27ae60',
                            text=[f"{r['calidad_historica_pct']}%" for r in campesinos],
                            textposition='outside'
                        ))
                    
                    if transportadores:
                        fig.add_trace(go.Bar(
                            name='🚛 Transportadores',
                            x=[r['vendedor'] for r in transportadores],
                            y=[r['calidad_historica_pct'] for r in transportadores],
                            marker_color='#3498db',
                            text=[f"{r['calidad_historica_pct']}%" for r in transportadores],
                            textposition='outside'
                        ))
                    
                    fig.update_layout(
                        title="Ranking de Calidad Histórica",
                        xaxis_title="Proveedor",
                        yaxis_title="Calidad (%)",
                        barmode='group',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detalles de cada proveedor
                    st.markdown("---")
                    for i, r in enumerate(recomendaciones, 1):
                        badge_class = "badge-campesino" if r['tipo'] == "Campesino" else "badge-transportador"
                        icon = "🌾" if r['tipo'] == "Campesino" else "🚛"
                        
                        with st.expander(f"#{i} - {r['vendedor']} ({r['tipo']}) - ⭐ {r['score']:.0f}/100"):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("📊 Calidad Histórica", f"{r['calidad_historica_pct']}%")
                                st.metric("📦 Total Vendidos", r['total_productos'])
                            
                            with col2:
                                st.metric("⚖️ Disponibles Ahora", r['productos_disponibles'])
                                st.metric("💰 Precio Promedio", f"${r['precio_promedio']:.2f}/kg")
                            
                            with col3:
                                st.metric("⭐ Score Global", f"{r['score']:.0f}/100")

    # ═══════════════════════════════════════════════════════════════════════
    #  TAB 3: REALIZAR COMPRA 
    # ═══════════════════════════════════════════════════════════════════════
    with tabs[2]:
        st.subheader("🛍️ Carrito de Compras")
        
        # Filtrar productos disponibles: Pendientes O Recogidos por transportistas
        df_disponibles = df_productos[
            (df_productos['estado'] == 'Pendiente') |
            ((df_productos['estado'] == 'Recogido') & (df_productos['origen'] == 'Transportador'))
        ]
        
        if df_disponibles.empty:
            st.warning("⚠️ No hay productos disponibles para compra.")
        else:
            # Crear lista de productos con información del vendedor
            opciones_productos = []
            indices_productos = []
            
            for idx, row in df_disponibles.iterrows():
                vendedor = row['campesino'] if row['origen'] == "Campesino" else row.get('transportador', row.get('transportista_asignado', 'Transportador'))
                tipo = "🌾" if row['origen'] == "Campesino" else "🚛"
                estado_info = " (En finca)" if row['origen'] == "Campesino" else " (Ya recogido - Disponible)"
                opciones_productos.append(
                    f"{row['producto']} - {row['ciudad']} - {tipo} {vendedor}{estado_info} (${row['precio_predicho']:.2f}/kg)"
                )
                indices_productos.append(idx)
            
            producto_compra = st.selectbox("🔍 Selecciona el producto:", opciones_productos, key="compra_producto")
            
            if producto_compra:
                idx_seleccionado = opciones_productos.index(producto_compra)
                producto_data = df_disponibles.iloc[idx_seleccionado]
                idx_real = indices_productos[idx_seleccionado]
                
                # Calcular reputación del vendedor
                vendedor_nombre = producto_data['campesino'] if producto_data['origen']=='Campesino' else producto_data.get('transportador', producto_data.get('transportista_asignado', 'Transportador'))
                rep_vendedor = calcular_reputacion(vendedor_nombre, df_productos)
                
                # Determinar estado del producto
                estado_producto = "En finca del campesino" if producto_data['origen'] == 'Campesino' else "Ya recogido - Listo para entrega"
                icono_estado = "🌾" if producto_data['origen'] == 'Campesino' else "🚛"
                
                # Card del producto seleccionado
                st.markdown(f"""
                <div class="producto-card">
                    <h3>📋 Detalles del Producto</h3>
                    <hr>
                    <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 15px;'>
                        <div>
                            <p><b>🌾 Producto:</b> {producto_data['producto']}</p>
                            <p><b>📍 Ciudad:</b> {producto_data['ciudad']}</p>
                            <p><b>👤 Vendedor:</b> {vendedor_nombre}</p>
                            <p><b>🏷️ Tipo:</b> <span class="{'badge-campesino' if producto_data['origen']=='Campesino' else 'badge-transportador'}">{producto_data['origen']}</span></p>
                            <p><b>{icono_estado} Estado:</b> {estado_producto}</p>
                        </div>
                        <div>
                            <p><b>⚖️ Disponible:</b> {producto_data['cantidad_kg']} kg</p>
                            <p><b>💰 Precio:</b> ${producto_data['precio_predicho']:.2f}/kg</p>
                            <p><b>🏆 Calidad:</b> {producto_data['calidad']}</p>
                            <p><b>⭐ Reputación:</b> {rep_vendedor['score']:.0f}/100</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Formulario de compra
                st.markdown("---")
                st.markdown("### 🛒 Datos de Compra")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    cantidad_compra = st.number_input(
                        "Cantidad (kg):", 
                        min_value=1.0, 
                        max_value=float(producto_data['cantidad_kg']), 
                        value=min(10.0, float(producto_data['cantidad_kg'])),
                        step=1.0,
                        key="cantidad_compra"
                    )
                
                with col2:
                    total_compra = cantidad_compra * producto_data['precio_predicho']
                    st.metric("💵 Total a Pagar", f"${total_compra:,.2f}")
                
                with col3:
                    ahorro_estimado = (producto_data['precio_predicho'] * 0.15) * cantidad_compra
                    st.metric("💚 Ahorro vs Retail", f"${ahorro_estimado:,.2f}", delta=f"-15%")
                
                # Ventaja de entrega inmediata
                if producto_data['origen'] == 'Transportador':
                    st.success("⚡ **ENTREGA INMEDIATA** - Este producto ya fue recogido por el transportador y está listo para entrega")
                else:
                    st.info("📍 Este producto requiere coordinación de transporte desde la finca")
                
                # Calificación y comentario (opcional)
                st.markdown("---")
                st.markdown("### ⭐ Evaluación del Producto (Opcional)")
                
                col_calif1, col_calif2 = st.columns([1, 2])
                
                with col_calif1:
                    calificacion = st.select_slider(
                        "Calificación esperada:",
                        options=[1, 2, 3, 4, 5],
                        value=5,
                        key="calificacion_compra"
                    )
                    st.write("⭐" * calificacion)
                
                with col_calif2:
                    comentario = st.text_area(
                        "Comentarios o instrucciones especiales:",
                        placeholder="Ej: Prefiero productos orgánicos, entrega en la mañana...",
                        key="comentario_compra",
                        height=100
                    )
                
                # Botón de confirmación
                st.markdown("---")
                if st.button("✅ Confirmar Compra", type="primary", use_container_width=True, key="btn_confirmar"):
                    if not nombre_comprador or nombre_comprador == "Comprador":
                        st.error("❌ Por favor ingresa tu nombre en la barra lateral.")
                    else:
                        compra_data = {
                            'origen': producto_data['origen'],
                            'campesino': producto_data['campesino'],
                            'transportador': producto_data.get('transportador', producto_data.get('transportista_asignado')),
                            'producto': producto_data['producto'],
                            'ciudad': producto_data['ciudad'],
                            'cantidad_kg': cantidad_compra,
                            'precio_unitario': producto_data['precio_predicho'],
                            'calificacion': calificacion,
                            'comentario': comentario
                        }
                        
                        if registrar_compra(nombre_comprador, compra_data):
                            # Actualizar estado del producto
                            # Si es de transportador (ya recogido), marcar como Vendido
                            # Si es de campesino (pendiente), marcar como Completado
                            if producto_data['origen'] == 'Transportador':
                                df_productos.loc[idx_real, 'estado'] = 'Vendido'
                            else:
                                df_productos.loc[idx_real, 'estado'] = 'Completado'
                            
                            # Si no se compró todo, ajustar cantidad o crear nuevo registro
                            if cantidad_compra < producto_data['cantidad_kg']:
                                cantidad_restante = producto_data['cantidad_kg'] - cantidad_compra
                                df_productos.loc[idx_real, 'cantidad_kg'] = cantidad_restante
                                # Mantener el estado según el origen
                                if producto_data['origen'] == 'Transportador':
                                    df_productos.loc[idx_real, 'estado'] = 'Recogido'  # Sigue disponible
                                else:
                                    df_productos.loc[idx_real, 'estado'] = 'Pendiente'  # Sigue en finca
                            
                            guardar_notificaciones(df_productos)
                            
                            st.success(f"✅ ¡Compra realizada con éxito!")
                            st.balloons()
                            
                            # Mostrar resumen
                            st.markdown(f"""
                            <div class="info-comprador">
                                <h3>🎉 ¡Gracias por tu compra!</h3>
                                <p><b>📦 Producto:</b> {producto_data['producto']}</p>
                                <p><b>⚖️ Cantidad:</b> {cantidad_compra} kg</p>
                                <p><b>💰 Total:</b> ${total_compra:,.2f}</p>
                                <p><b>👤 Vendedor:</b> {vendedor_nombre}</p>
                                <hr>
                                <p>📱 El vendedor será notificado y te contactará pronto para coordinar la entrega.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.error("❌ Error al procesar la compra. Intenta nuevamente.")

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 4: ANÁLISIS DE PRECIOS 
    # ═══════════════════════════════════════════════════════════════════════
    with tabs[3]:
        st.subheader("📈 Análisis de Tendencias y Mejores Precios")
        
        productos_analisis = sorted(df_productos['producto'].dropna().unique().tolist())
        ciudades_analisis = sorted(df_productos['ciudad'].dropna().unique().tolist())
        
        if not productos_analisis or not ciudades_analisis:
            st.info("📭 No hay suficientes datos históricos para análisis.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                producto_analisis = st.selectbox("Producto:", productos_analisis, key="analisis_producto")
            
            with col2:
                ciudad_analisis = st.selectbox("Ciudad:", ciudades_analisis, key="analisis_ciudad")
            
            if st.button("📊 Generar Análisis", type="primary", use_container_width=True):
                tendencia = analizar_tendencia_precios(producto_analisis, ciudad_analisis, df_productos)
                
                if tendencia is None or tendencia.empty:
                    st.warning("⚠️ No hay suficientes datos históricos para este producto y ciudad.")
                else:
                    # Gráfico de tendencia de precios
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=tendencia['mes'],
                        y=tendencia['precio_promedio'],
                        mode='lines+markers',
                        name='Precio Promedio',
                        line=dict(color='#667eea', width=3),
                        marker=dict(size=10)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=tendencia['mes'],
                        y=tendencia['precio_max'],
                        mode='lines',
                        name='Precio Máximo',
                        line=dict(color='#e74c3c', width=2, dash='dash'),
                        fill=None
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=tendencia['mes'],
                        y=tendencia['precio_min'],
                        mode='lines',
                        name='Precio Mínimo',
                        line=dict(color='#2ecc71', width=2, dash='dash'),
                        fill='tonexty',
                        fillcolor='rgba(102, 126, 234, 0.2)'
                    ))
                    
                    fig.update_layout(
                        title=f"Tendencia de Precios - {producto_analisis} en {ciudad_analisis}",
                        xaxis_title="Mes",
                        yaxis_title="Precio ($/kg)",
                        hovermode='x unified',
                        height=450
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Sugerencia de mejor momento para comprar
                    sugerencia = sugerir_mejor_momento_compra(tendencia)
                    
                    if sugerencia:
                        st.markdown("---")
                        st.markdown("### 💡 Recomendación Inteligente")
                        
                        col_sug1, col_sug2, col_sug3 = st.columns(3)
                        
                        with col_sug1:
                            st.markdown(f"""
                            <div class="stat-card" style="background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%); color: white;">
                                <div class="stat-value" style="color: white;">💚 Mejor Mes</div>
                                <div class="stat-label" style="color: white; font-size: 20px; margin-top: 10px;">{sugerencia['mes_barato']}</div>
                                <p style="margin: 10px 0 0 0;">Precio: ${sugerencia['precio_barato']:.2f}/kg</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_sug2:
                            st.markdown(f"""
                            <div class="stat-card" style="background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); color: white;">
                                <div class="stat-value" style="color: white;">❌ Mes Más Caro</div>
                                <div class="stat-label" style="color: white; font-size: 20px; margin-top: 10px;">{sugerencia['mes_caro']}</div>
                                <p style="margin: 10px 0 0 0;">Precio: ${sugerencia['precio_caro']:.2f}/kg</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_sug3:
                            st.markdown(f"""
                            <div class="stat-card" style="background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%); color: white;">
                                <div class="stat-value" style="color: white;">💰 Ahorro</div>
                                <div class="stat-label" style="color: white; font-size: 20px; margin-top: 10px;">{sugerencia['ahorro_pct']}%</div>
                                <p style="margin: 10px 0 0 0;">Comprando en {sugerencia['mes_barato']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.info(f"💡 **Consejo:** Comprar {producto_analisis} en {sugerencia['mes_barato']} puede ahorrarte hasta {sugerencia['ahorro_pct']}% comparado con {sugerencia['mes_caro']}.")
                    
                    # Tabla de datos históricos
                    st.markdown("---")
                    st.markdown("### 📋 Datos Históricos Detallados")
                    
                    tendencia_display = tendencia.copy()
                    tendencia_display.columns = ['Mes', 'Precio Promedio', 'Precio Mínimo', 'Precio Máximo', 'Registros']
                    tendencia_display['Precio Promedio'] = tendencia_display['Precio Promedio'].apply(lambda x: f"${x:.2f}")
                    tendencia_display['Precio Mínimo'] = tendencia_display['Precio Mínimo'].apply(lambda x: f"${x:.2f}")
                    tendencia_display['Precio Máximo'] = tendencia_display['Precio Máximo'].apply(lambda x: f"${x:.2f}")
                    
                    st.dataframe(tendencia_display, use_container_width=True, hide_index=True)

    # ═══════════════════════════════════════════════════════════════════════
    #  TAB 5: MIS ALERTAS 
    # ═══════════════════════════════════════════════════════════════════════
    with tabs[4]:
        st.subheader("🔔 Sistema de Alertas de Precio")
        
        st.markdown("""
        <div class="info-comprador">
            <h4 style='margin-top:0;'>¿Cómo funciona?</h4>
            <p>Configura alertas para recibir notificaciones cuando un producto alcance tu precio objetivo.</p>
            <p>Te avisaremos automáticamente cuando haya productos disponibles a tu precio deseado.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Crear nueva alerta
        st.markdown("### ➕ Crear Nueva Alerta")
        
        with st.form("form_nueva_alerta"):
            col1, col2, col3 = st.columns(3)
            
            productos_alerta = sorted(df_productos['producto'].dropna().unique().tolist())
            ciudades_alerta = sorted(df_productos['ciudad'].dropna().unique().tolist())
            
            with col1:
                producto_alerta = st.selectbox("Producto:", productos_alerta, key="alerta_producto")
            
            with col2:
                ciudad_alerta = st.selectbox("Ciudad:", ciudades_alerta, key="alerta_ciudad")
            
            with col3:
                precio_objetivo = st.number_input(
                    "Precio objetivo ($/kg):",
                    min_value=100,
                    max_value=100000,
                    value=5000,
                    step=100,
                    key="alerta_precio"
                )
            
            submitted_alerta = st.form_submit_button("✅ Crear Alerta", use_container_width=True)
            
            if submitted_alerta:
                nueva_alerta = {
                    'comprador': nombre_comprador,
                    'producto': producto_alerta,
                    'ciudad': ciudad_alerta,
                    'precio_objetivo': precio_objetivo,
                    'activa': True,
                    'fecha_creacion': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                df_alertas = pd.concat([df_alertas, pd.DataFrame([nueva_alerta])], ignore_index=True)
                guardar_alertas(df_alertas)
                
                st.success(f"✅ Alerta creada: Te notificaremos cuando {producto_alerta} en {ciudad_alerta} esté a ${precio_objetivo}/kg o menos.")
                st.rerun()
        
        # Mostrar alertas activas
        st.markdown("---")
        st.markdown("### 📋 Mis Alertas Activas")
        
        alertas_usuario = df_alertas[
            (df_alertas['comprador'] == nombre_comprador) & 
            (df_alertas['activa'] == True)
        ]
        
        if alertas_usuario.empty:
            st.info("📭 No tienes alertas activas. ¡Crea una arriba!")
        else:
            for idx, alerta in alertas_usuario.iterrows():
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    st.markdown(f"""
                    <div class="producto-card">
                        <h4>{alerta['producto']} en {alerta['ciudad']}</h4>
                        <p><b>🎯 Precio objetivo:</b> ${alerta['precio_objetivo']:.2f}/kg</p>
                        <p><b>📅 Creada:</b> {alerta['fecha_creacion']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    if st.button("🗑️", key=f"eliminar_alerta_{idx}"):
                        df_alertas.loc[idx, 'activa'] = False
                        guardar_alertas(df_alertas)
                        st.success("✅ Alerta desactivada")
                        st.rerun()

    # ═══════════════════════════════════════════════════════════════════════
    #  TAB 6: MI HISTORIAL 
    # ═══════════════════════════════════════════════════════════════════════
    with tabs[5]:
        st.subheader("📜 Historial de Compras")
        
        if df_compras.empty:
            st.info("📭 Aún no has realizado compras.")
        else:
            compras_usuario = df_compras[df_compras['comprador'] == nombre_comprador]
            
            if compras_usuario.empty:
                st.info("📭 No tienes compras registradas con este nombre.")
            else:
                # Estadísticas generales
                st.markdown("### 📊 Resumen General")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-value">{len(compras_usuario)}</div>
                        <div class="stat-label">Total Compras</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    total_gastado = compras_usuario['precio_total'].sum()
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-value">${total_gastado:,.0f}</div>
                        <div class="stat-label">Total Gastado</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    total_kg = compras_usuario['cantidad_kg'].sum()
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-value">{total_kg:.0f} kg</div>
                        <div class="stat-label">Total Comprado</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    productos_unicos = compras_usuario['producto'].nunique()
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-value">{productos_unicos}</div>
                        <div class="stat-label">Productos Distintos</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Gráfico de compras por producto
                st.markdown("---")
                st.markdown("### 📊 Mis Productos Más Comprados")
                
                compras_por_producto = compras_usuario.groupby('producto').agg({
                    'cantidad_kg': 'sum',
                    'precio_total': 'sum'
                }).reset_index().sort_values('cantidad_kg', ascending=False)
                
                fig = px.bar(
                    compras_por_producto,
                    x='producto',
                    y='cantidad_kg',
                    title="Cantidad Comprada por Producto (kg)",
                    labels={'producto': 'Producto', 'cantidad_kg': 'Cantidad (kg)'},
                    color='cantidad_kg',
                    color_continuous_scale='Viridis'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Historial detallado
                st.markdown("---")
                st.markdown("### 📋 Detalle de Compras")
                
                compras_ordenadas = compras_usuario.sort_values('fecha_compra', ascending=False)
                
                for idx, compra in compras_ordenadas.iterrows():
                    with st.expander(f"🛍️ {compra['producto']} - {compra['fecha_compra']} - ${compra['precio_total']:,.2f}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**📦 Producto:** {compra['producto']}")
                            st.write(f"**👤 Vendedor:** {compra['vendedor']}")
                            st.write(f"**🏷️ Tipo:** {compra['origen']}")
                            st.write(f"**📍 Ciudad:** {compra['ciudad']}")
                        
                        with col2:
                            st.write(f"**⚖️ Cantidad:** {compra['cantidad_kg']} kg")
                            st.write(f"**💰 Precio/kg:** ${compra['precio_unitario']:,.2f}")
                            st.write(f"**💵 Total:** ${compra['precio_total']:,.2f}")
                            
                            if pd.notna(compra.get('calificacion')) and compra['calificacion'] > 0:
                                st.write(f"**⭐ Calificación:** {'⭐' * int(compra['calificacion'])}")
                        
                        if pd.notna(compra.get('comentario')) and compra['comentario']:
                            st.write(f"**💬 Comentario:** {compra['comentario']}")
#EJECUCIÓN DIRECTA
if __name__ == "__main__":
    st.set_page_config(
        page_title="Marketplace Agrícola - Comprador",
        page_icon="🛒",
        layout="wide"
    )
    view_comprador()