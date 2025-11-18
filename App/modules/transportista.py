import streamlit as st
import pandas as pd
import os
from datetime import datetime
import numpy as np
import folium
from streamlit_folium import st_folium
from folium import plugins
from itertools import permutations
import time


# CONFIGURACIÓN

APP_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.join(APP_ROOT, "data")
CSV_NOTIFICACIONES = os.path.join(DATA_DIR, "notificaciones_transporte.csv")
CSV_VENTAS_TRANSPORTADOR = os.path.join(DATA_DIR, "ventas_transportador.csv")
os.makedirs(DATA_DIR, exist_ok=True)

UBICACIONES_CIUDADES = {
    'Tunja': (5.5353, -73.3678),
    'Duitama': (5.8269, -73.0347),
    'Sogamoso': (5.7147, -72.9342),
    'Paipa': (5.7808, -73.1175),
    'Chiquinquirá': (5.6181, -73.8169),
    'Villa de Leyva': (5.6378, -73.5264),
    'Nobsa': (5.7703, -72.9486),
    'Tibasosa': (5.7506, -72.9828),
    'Moniquirá': (5.8753, -73.5750),
    'Samacá': (5.4892, -73.4956)
}


# FUNCIONES DE DATOS

def cargar_notificaciones():
    if os.path.exists(CSV_NOTIFICACIONES):
        df = pd.read_csv(CSV_NOTIFICACIONES)
        columnas_necesarias = [
            'transportista_lat', 'transportista_lon', 'distancia_restante_km',
            'progreso_viaje', 'ruta_optimizada', 'orden_parada',
            'tiempo_estimado_llegada', 'notificacion_enviada', 'telefono_campesino',
            'origen', 'transportador'
        ]
        for col in columnas_necesarias:
            if col not in df.columns:
                df[col] = None
        return df
    return pd.DataFrame(columns=[
        'id_notificacion', 'fecha_notificacion', 'campesino', 'producto',
        'cantidad_kg', 'ciudad', 'direccion', 'precio', 'precio_predicho',
        'calidad', 'estado', 'fecha_recogida', 'transportista_asignado',
        'imagen', 'latitud', 'longitud', 'transportista_lat', 'transportista_lon',
        'distancia_restante_km', 'progreso_viaje', 'ruta_optimizada', 'orden_parada',
        'tiempo_estimado_llegada', 'notificacion_enviada', 'telefono_campesino',
        'origen', 'transportador'
    ])

def guardar_notificaciones(df):
    try:
        df.to_csv(CSV_NOTIFICACIONES, index=False)
        return True
    except Exception as e:
        st.error(f"Error al guardar: {e}")
        return False

def validar_coordenadas(lat, lon):
    return lat is not None and lon is not None and not pd.isna(lat) and not pd.isna(lon)

def agrupar_por_proximidad(df_disponibles, radio_km=5):
    """Agrupa productos que están cerca unos de otros"""
    if df_disponibles.empty:
        return []
    
    df_validos = df_disponibles[
        df_disponibles.apply(lambda row: validar_coordenadas(row['latitud'], row['longitud']), axis=1)
    ].copy()
    
    if df_validos.empty:
        return []
    
    grupos = []
    indices_procesados = set()
    
    for idx1, row1 in df_validos.iterrows():
        if idx1 in indices_procesados:
            continue
        
        grupo_actual = {
            'indices': [idx1],
            'productos': [row1],
            'centro': (row1['latitud'], row1['longitud']),
            'ciudad': row1['ciudad']
        }
        indices_procesados.add(idx1)
        
        for idx2, row2 in df_validos.iterrows():
            if idx2 in indices_procesados:
                continue
            
            if row1['ciudad'] == row2['ciudad']:
                distancia = calcular_distancia(
                    (row1['latitud'], row1['longitud']),
                    (row2['latitud'], row2['longitud'])
                )
                
                if distancia <= radio_km:
                    grupo_actual['indices'].append(idx2)
                    grupo_actual['productos'].append(row2)
                    indices_procesados.add(idx2)
        
        grupos.append(grupo_actual)
    
    grupos.sort(key=lambda g: len(g['productos']), reverse=True)
    return grupos

def calcular_distancia(coord1, coord2):
    R = 6371
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
    c = 2*np.arcsin(np.sqrt(a))
    return R * c

def simular_movimiento(ciudad_origen, lat_destino, lon_destino, progreso_actual=0.0):
    lat_origen, lon_origen = UBICACIONES_CIUDADES.get(ciudad_origen, UBICACIONES_CIUDADES['Tunja'])
    incremento = np.random.uniform(0.05, 0.15)
    nuevo_progreso = min(progreso_actual + incremento, 1.0)
    lat_actual = lat_origen + (lat_destino - lat_origen) * nuevo_progreso
    lon_actual = lon_origen + (lon_destino - lon_origen) * nuevo_progreso
    if nuevo_progreso >= 1.0:
        lat_actual, lon_actual = lat_destino, lon_destino
    distancia_restante = calcular_distancia((lat_actual, lon_actual), (lat_destino, lon_destino))
    tiempo_minutos = (distancia_restante / 40) * 60
    return lat_actual, lon_actual, nuevo_progreso, distancia_restante, tiempo_minutos

def optimizar_ruta_ia(origen, destinos):
    if len(destinos) <= 1:
        return destinos, sum([calcular_distancia(origen, (d['lat'], d['lon'])) for d in destinos])
    if len(destinos) <= 8:
        mejor_ruta = None
        mejor_distancia = float('inf')
        for perm in permutations(destinos):
            distancia_total = calcular_distancia(origen, (perm[0]['lat'], perm[0]['lon']))
            for i in range(len(perm)-1):
                distancia_total += calcular_distancia((perm[i]['lat'], perm[i]['lon']), (perm[i+1]['lat'], perm[i+1]['lon']))
            if distancia_total < mejor_distancia:
                mejor_distancia = distancia_total
                mejor_ruta = list(perm)
        return mejor_ruta, mejor_distancia
    # heurística vecino más cercano
    ruta = []
    pendientes = destinos.copy()
    posicion_actual = origen
    distancia_total = 0
    while pendientes:
        distancias = [calcular_distancia(posicion_actual, (d['lat'], d['lon'])) for d in pendientes]
        idx_cercano = np.argmin(distancias)
        destino = pendientes.pop(idx_cercano)
        distancia_total += distancias[idx_cercano]
        ruta.append(destino)
        posicion_actual = (destino['lat'], destino['lon'])
    return ruta, distancia_total

def crear_mapa(ciudad_origen, viajes_activos):
    lat_origen, lon_origen = UBICACIONES_CIUDADES.get(ciudad_origen, UBICACIONES_CIUDADES['Tunja'])
    mapa = folium.Map(location=[lat_origen, lon_origen], zoom_start=11, control_scale=True)
    plugins.Fullscreen(position='topright').add_to(mapa)
    
    folium.Marker([lat_origen, lon_origen], popup=f"🏢 Base: {ciudad_origen}", tooltip='Base', icon=folium.Icon(color='blue', icon='home', prefix='fa')).add_to(mapa)
    
    colores_ruta = ['#4285F4', '#EA4335', '#FBBC04', '#34A853', '#FF6D00', '#9C27B0']
    
    if not viajes_activos.empty and 'ruta_optimizada' in viajes_activos.columns:
        rutas = viajes_activos.groupby('ruta_optimizada', dropna=False)
        
        for i, (nombre_ruta, grupo_ruta) in enumerate(rutas):
            color_actual = colores_ruta[i % len(colores_ruta)]
            grupo_ruta = grupo_ruta.sort_values('orden_parada')
            
            puntos_ruta = [[lat_origen, lon_origen]]
            
            for idx, row in grupo_ruta.iterrows():
                if not validar_coordenadas(row['latitud'], row['longitud']):
                    continue
                
                lat_dest, lon_dest = row['latitud'], row['longitud']
                lat_trans, lon_trans = row.get('transportista_lat'), row.get('transportista_lon')
                orden = row.get('orden_parada', 1)
                
                if not validar_coordenadas(lat_trans, lon_trans):
                    lat_trans, lon_trans = lat_origen, lon_origen
                
                puntos_ruta.append([lat_dest, lon_dest])
                
                folium.Marker(
                    [lat_dest, lon_dest],
                    popup=f"📍 Parada {orden}: {row['campesino']}<br>Producto: {row['producto']}<br>Progreso: {row.get('progreso_viaje',0)*100:.0f}%",
                    tooltip=f"Parada {orden}",
                    icon=folium.DivIcon(html=f"""
                        <div style="font-size: 20px; color: white; background-color: {color_actual}; 
                                    border-radius: 50%; width: 30px; height: 30px; 
                                    display: flex; align-items: center; justify-content: center;
                                    border: 2px solid white; font-weight: bold;">
                            {orden}
                        </div>
                    """)
                ).add_to(mapa)
                
                if orden == 1 or nombre_ruta == "Entrega Individual":
                    folium.Marker(
                        [lat_trans, lon_trans],
                        popup=f"🚚 Transportista<br>Ruta: {nombre_ruta}<br>Progreso: {row.get('progreso_viaje',0)*100:.0f}%",
                        icon=folium.Icon(color='red', icon='truck', prefix='fa')
                    ).add_to(mapa)
            
            folium.PolyLine(puntos_ruta, color=color_actual, weight=4, opacity=0.7, 
                          popup=f"Ruta: {nombre_ruta}").add_to(mapa)
    
    return mapa

def aplicar_estilos():
    st.markdown("""
    <style>
    .stat-card {background:white; padding:15px; border-radius:12px; text-align:center; border-left:4px solid #4285F4; box-shadow:0 2px 8px rgba(0,0,0,0.1);}
    .stat-value {font-size:28px; font-weight:bold; color:#4285F4;}
    .stat-label {font-size:12px; color:#5f6368; text-transform:uppercase;}
    .producto-venta {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 12px;
        margin: 10px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)


# VISTA TRANSPORTISTA

def view_transportista():
    aplicar_estilos()
    
    st.markdown("<h1 style='text-align:center;'>🚚 AgroMov Pro</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#5f6368;'>Encuentra cargas • Optimiza rutas • Gestiona entregas • Vende productos</p>", unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### 👤 Transportista")
        nombre_transportista = st.text_input("Nombre", "Carlos Pérez")
        ciudad_origen = st.selectbox("🏙️ Ciudad Base", list(UBICACIONES_CIUDADES.keys()))
        st.markdown("---")
        
        st.markdown("### 🎯 Configuración de Rutas")
        agrupar_entregas = st.checkbox("📦 Agrupar entregas cercanas", value=True)
        if agrupar_entregas:
            radio_agrupacion = st.slider("Radio de agrupación (km)", 1, 15, 5)
            st.info(f"🗺️ Se agruparán productos dentro de {radio_agrupacion} km")
        
        st.markdown("---")
        auto_update = st.checkbox("🔄 Actualización automática", value=False)
        if auto_update:
            intervalo = st.slider("Intervalo (segundos)", 2, 10, 5)
            st.info(f"📍 Las ubicaciones se actualizarán cada {intervalo} segundos automáticamente.")
        else:
            st.info("📍 Usa el botón 'Actualizar ubicación' para simular movimiento hacia el destino.")
    
    df_notif = cargar_notificaciones()

    # TABS PRINCIPALES
    tabs = st.tabs(["🎯 Cargas Disponibles", "🚛 Entregas en Curso", "🛒 Mis Productos en Venta", "🗺️ Mapa", "📊 Estadísticas"])

    #  TAB 1: CARGAS DISPONIBLES 
    with tabs[0]:
        notif_disponibles = df_notif[df_notif['estado'] == 'Pendiente']

        st.markdown("### 🎯 Cargas Disponibles para Aceptar")

        if notif_disponibles.empty:
            st.info("No hay cargas pendientes para recoger en este momento.")
        else:
            if agrupar_entregas:
                grupos = agrupar_por_proximidad(notif_disponibles, radio_agrupacion)
                
                st.markdown(f"**Se encontraron {len(grupos)} zona(s) con productos disponibles**")
                
                for i, grupo in enumerate(grupos, 1):
                    num_productos = len(grupo['productos'])
                    total_kg = sum([p['cantidad_kg'] for p in grupo['productos']])
                    total_precio = sum([p['precio'] for p in grupo['productos']])
                    
                    with st.expander(f"🗺️ Zona {i} - {grupo['ciudad']} ({num_productos} producto{'s' if num_productos > 1 else ''}) - ${total_precio:,.0f}", expanded=(i==1)):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("📦 Productos", num_productos)
                        with col2:
                            st.metric("⚖️ Total kg", f"{total_kg:.0f}")
                        with col3:
                            st.metric("💰 Ingreso Total", f"${total_precio:,.0f}")
                        
                        st.markdown("---")
                        
                        lat_o, lon_o = UBICACIONES_CIUDADES.get(ciudad_origen, UBICACIONES_CIUDADES['Tunja'])
                        destinos = [{'lat': p['latitud'], 'lon': p['longitud'], 'nombre': p['campesino'], 'producto': p['producto']} 
                                   for p in grupo['productos']]
                        ruta_optimizada, distancia_total = optimizar_ruta_ia((lat_o, lon_o), destinos)
                        
                        st.markdown(f"""
                        <div style='background:#e8f5e9; padding:12px; border-radius:8px; margin:10px 0;'>
                            <h5>🚀 Ruta Optimizada por IA</h5>
                            <p><b>📏 Distancia total:</b> {distancia_total:.2f} km</p>
                            <p><b>⏱️ Tiempo estimado:</b> {(distancia_total/40)*60:.0f} minutos</p>
                            <p><b>🔄 Orden de recogida:</b></p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        for j, destino in enumerate(ruta_optimizada, 1):
                            st.markdown(f"""
                            <div style='background:#f8f9fa; padding:10px; border-radius:8px; margin:8px 0;
                                        border-left:3px solid #34A853;'>
                                <h5>📍 Parada {j}: {destino['producto']}</h5>
                                <p><b>Campesino:</b> {destino['nombre']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("---")
                        
                        if st.button(f"✅ Aceptar todas las entregas de Zona {i}", key=f"aceptar_grupo_{i}"):
                            for orden, destino in enumerate(ruta_optimizada, 1):
                                idx = None
                                for k, p in zip(grupo['indices'], grupo['productos']):
                                    if p['campesino'] == destino['nombre'] and p['producto'] == destino['producto']:
                                        idx = k
                                        break
                                
                                if idx is not None:
                                    df_notif.loc[idx, 'estado'] = 'Aceptado'
                                    df_notif.loc[idx, 'transportista_asignado'] = nombre_transportista
                                    df_notif.loc[idx, 'progreso_viaje'] = 0.0
                                    df_notif.loc[idx, 'transportista_lat'] = lat_o
                                    df_notif.loc[idx, 'transportista_lon'] = lon_o
                                    df_notif.loc[idx, 'ruta_optimizada'] = f"Zona_{i}"
                                    df_notif.loc[idx, 'orden_parada'] = orden
                            
                            guardar_notificaciones(df_notif)
                            st.success(f"✅ Has aceptado {num_productos} entregas en Zona {i}. Distancia total: {distancia_total:.1f} km, Tiempo: {(distancia_total/40)*60:.0f} min")
                            st.rerun()
            else:
                for idx, row in notif_disponibles.iterrows():
                    if not validar_coordenadas(row['latitud'], row['longitud']):
                        continue

                    st.markdown(f"""
                    <div style='background:white; padding:15px; border-radius:10px; margin:10px 0;
                                box-shadow:0 2px 6px rgba(0,0,0,0.1); border-left:4px solid #34A853;'>
                        <h4>🌾 {row['producto']}</h4>
                        <p><b>Campesino:</b> {row['campesino']}</p>
                        <p><b>Ciudad:</b> {row['ciudad']}</p>
                        <p><b>Cantidad:</b> {row['cantidad_kg']} kg</p>
                        <p><b>Precio:</b> ${row['precio']:,.0f}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    if st.button(f"✅ Aceptar {row['producto']}", key=f"aceptar_{idx}"):
                        lat_o, lon_o = UBICACIONES_CIUDADES.get(ciudad_origen, UBICACIONES_CIUDADES['Tunja'])
                        df_notif.loc[idx, 'estado'] = 'Aceptado'
                        df_notif.loc[idx, 'transportista_asignado'] = nombre_transportista
                        df_notif.loc[idx, 'progreso_viaje'] = 0.0
                        df_notif.loc[idx, 'transportista_lat'] = lat_o
                        df_notif.loc[idx, 'transportista_lon'] = lon_o
                        df_notif.loc[idx, 'orden_parada'] = 1
                        guardar_notificaciones(df_notif)
                        st.success(f"✅ Has aceptado recoger {row['producto']} de {row['campesino']}")
                        st.rerun()

    # TAB 2: ENTREGAS EN CURSO
    with tabs[1]:
        viajes_activos = df_notif[
            (df_notif['transportista_asignado'] == nombre_transportista) &
            (df_notif['estado'] == 'Aceptado')
        ]

        if auto_update and not viajes_activos.empty:
            tiempo_actualizado = False
            for idx, row in viajes_activos.iterrows():
                progreso = row.get('progreso_viaje', 0.0)
                if progreso < 1.0:
                    lat_t, lon_t, prog, dist, tiempo_est = simular_movimiento(
                        ciudad_origen, row['latitud'], row['longitud'], progreso
                    )
                    df_notif.loc[idx, ['transportista_lat', 'transportista_lon',
                                       'progreso_viaje', 'distancia_restante_km',
                                       'tiempo_estimado_llegada']] = [lat_t, lon_t, prog, dist, tiempo_est]
                    tiempo_actualizado = True
            
            if tiempo_actualizado:
                guardar_notificaciones(df_notif)
                time.sleep(intervalo)
                st.rerun()

        if viajes_activos.empty:
            st.info("No tienes entregas activas en este momento.")
        else:
            st.markdown("### 🚛 Mis Entregas en Curso")

            rutas = viajes_activos.groupby('ruta_optimizada', dropna=False)
            
            for nombre_ruta, grupo_ruta in rutas:
                if pd.isna(nombre_ruta):
                    nombre_ruta = "Entrega Individual"
                
                grupo_ruta = grupo_ruta.sort_values('orden_parada')
                num_paradas = len(grupo_ruta)
                
                st.markdown(f"#### 🗺️ {nombre_ruta} ({num_paradas} parada{'s' if num_paradas > 1 else ''})")
                
                for idx, row in grupo_ruta.iterrows():
                    progreso = row.get('progreso_viaje', 0.0)
                    distancia = row.get('distancia_restante_km', 0)
                    tiempo = row.get('tiempo_estimado_llegada', 0)
                    orden = row.get('orden_parada', 1)

                    st.markdown(f"""
                    <div style='background:white; padding:15px; border-radius:10px; margin:10px 0;
                                box-shadow:0 2px 6px rgba(0,0,0,0.1); border-left:4px solid #EA4335;'>
                        <h4>📍 Parada {orden}: {row['producto']} - {row['campesino']}</h4>
                        <p><b>Ciudad:</b> {row['ciudad']} | <b>Cantidad:</b> {row['cantidad_kg']} kg</p>
                        <p><b>Progreso:</b> {progreso*100:.1f}% | <b>Distancia restante:</b> {distancia:.1f} km | <b>ETA:</b> {tiempo:.0f} min</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(progreso)

                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col1:
                        if st.button(f"📍 Actualizar ubicación", key=f"ubicacion_{idx}"):
                            lat_t, lon_t, prog, dist, tiempo = simular_movimiento(
                                ciudad_origen, row['latitud'], row['longitud'], progreso
                            )
                            df_notif.loc[idx, ['transportista_lat', 'transportista_lon',
                                               'progreso_viaje', 'distancia_restante_km',
                                               'tiempo_estimado_llegada']] = [lat_t, lon_t, prog, dist, tiempo]
                            guardar_notificaciones(df_notif)
                            st.success("📍 Ubicación actualizada.")
                            st.rerun()

                    with col2:
                        if progreso >= 0.95:
                            if st.button(f"✅ Marcar como Recogido", key=f"recogido_{idx}"):
                                # IMPORTANTE: Marcar como recogido Y asignar al transportista como vendedor
                                df_notif.loc[idx, 'estado'] = 'Recogido'
                                df_notif.loc[idx, 'fecha_recogida'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                df_notif.loc[idx, 'origen'] = 'Transportador'  # Cambiar origen
                                df_notif.loc[idx, 'transportador'] = nombre_transportista  # Asignar como vendedor
                                guardar_notificaciones(df_notif)
                                st.success("✅ Producto recogido. Ahora disponible para venta.")
                                st.rerun()

                    with col3:
                        st.metric("Avance", f"{progreso*100:.1f}%")
                
                st.markdown("---")

    # TAB 3: PRODUCTOS EN VENTA
    with tabs[2]:
        st.markdown("### 🛒 Mis Productos Disponibles para Venta")
        
        productos_venta = df_notif[
            (df_notif['transportista_asignado'] == nombre_transportista) &
            (df_notif['estado'] == 'Recogido') &
            (df_notif['origen'] == 'Transportador')
        ]
        
        if productos_venta.empty:
            st.info("📦 No tienes productos recogidos para vender. Recoge productos primero en la sección 'Entregas en Curso'.")
        else:
            st.success(f"✅ Tienes **{len(productos_venta)}** producto(s) disponible(s) para venta")
            
            # Estadísticas de productos en venta
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_kg_venta = productos_venta['cantidad_kg'].sum()
                st.metric("⚖️ Total kg disponible", f"{total_kg_venta:.0f}")
            
            with col2:
                valor_total = productos_venta['precio_predicho'].fillna(productos_venta['precio']).sum()
                st.metric("💰 Valor Potencial", f"${valor_total:,.0f}")
            
            with col3:
                productos_vendidos = len(df_notif[
                    (df_notif['transportista_asignado'] == nombre_transportista) &
                    (df_notif['estado'] == 'Vendido')
                ])
                st.metric("✅ Ya vendidos", productos_vendidos)
            
            st.markdown("---")
            st.markdown("### 📦 Inventario de Productos")
            
            for idx, row in productos_venta.iterrows():
                st.markdown(f"""
                <div class="producto-venta">
                    <h3>🌾 {row['producto']}</h3>
                    <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px;'>
                        <div>
                            <p><b>📍 Ciudad:</b> {row['ciudad']}</p>
                            <p><b>⚖️ Cantidad:</b> {row['cantidad_kg']} kg</p>
                            <p><b>🏆 Calidad:</b> {row['calidad']}</p>
                        </div>
                        <div>
                            <p><b>💰 Precio:</b> ${row.get('precio_predicho', row['precio']):,.0f}/kg</p>
                            <p><b>👤 Origen:</b> {row['campesino']}</p>
                            <p><b>📅 Recogido:</b> {row['fecha_recogida']}</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.info(f"✅ Este producto está visible para compradores en el marketplace")
                with col_b:
                    if st.button("🗑️ Retirar", key=f"retirar_{idx}"):
                        df_notif.loc[idx, 'estado'] = 'Retirado'
                        guardar_notificaciones(df_notif)
                        st.success("Producto retirado del marketplace")
                        st.rerun()
                
                st.markdown("---")
            
            # Historial de ventas
            st.markdown("### 📊 Historial de Ventas")
            
            productos_vendidos_lista = df_notif[
                (df_notif['transportista_asignado'] == nombre_transportista) &
                (df_notif['estado'] == 'Vendido')
            ]
            
            if not productos_vendidos_lista.empty:
                st.success(f"✅ Has vendido {len(productos_vendidos_lista)} producto(s)")
                
                for idx, row in productos_vendidos_lista.iterrows():
                    st.markdown(f"""
                    <div style='background:#e8f5e9; padding:12px; border-radius:8px; margin:10px 0;'>
                        <h4>✅ {row['producto']} - {row['cantidad_kg']} kg</h4>
                        <p><b>💰 Precio:</b> ${row.get('precio_predicho', row['precio']):,.0f}</p>
                        <p><b>📅 Fecha:</b> {row.get('fecha_recogida', 'N/A')}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Aún no has vendido productos. Los productos marcados como 'Recogido' están disponibles para que los compradores los adquieran.")

    # TAB 4: MAPA 
    with tabs[3]:
        viajes_activos = df_notif[
            (df_notif['transportista_asignado'] == nombre_transportista) &
            (df_notif['estado'] == 'Aceptado')
        ]
        
        st.markdown("### 🗺️ Mapa de Operaciones en Tiempo Real")
        mapa = crear_mapa(ciudad_origen, viajes_activos)
        st_folium(mapa, width=1400, height=600)

    # TAB 5: ESTADÍSTICAS 
    with tabs[4]:
        st.markdown("### 📊 Panel de Estadísticas")
        
        # Calcular métricas generales
        viajes_activos = df_notif[
            (df_notif['transportista_asignado'] == nombre_transportista) &
            (df_notif['estado'] == 'Aceptado')
        ]
        
        productos_recogidos = df_notif[
            (df_notif['transportista_asignado'] == nombre_transportista) &
            (df_notif['estado'] == 'Recogido')
        ]
        
        productos_vendidos = df_notif[
            (df_notif['transportista_asignado'] == nombre_transportista) &
            (df_notif['estado'] == 'Vendido')
        ]
        
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            ingresos_activos = viajes_activos['precio'].sum() if not viajes_activos.empty else 0
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">${ingresos_activos:,.0f}</div>
                <div class="stat-label">Ingresos Potenciales</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            total_recogidos = len(productos_recogidos)
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{total_recogidos}</div>
                <div class="stat-label">Productos Recogidos</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            total_vendidos = len(productos_vendidos)
            ingresos_vendidos = productos_vendidos['precio_predicho'].fillna(productos_vendidos['precio']).sum() if not productos_vendidos.empty else 0
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{total_vendidos}</div>
                <div class="stat-label">Productos Vendidos</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">${ingresos_vendidos:,.0f}</div>
                <div class="stat-label">Ingresos Reales</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Progreso de entregas activas
        if not viajes_activos.empty:
            st.markdown("### 🚛 Progreso de Entregas Activas")
            promedio = viajes_activos['progreso_viaje'].mean() * 100
            
            st.metric("⚡ Progreso Promedio", f"{promedio:.1f}%")
            st.progress(promedio / 100)
            
            # Tabla de viajes activos
            st.markdown("#### 📋 Detalle de Viajes")
            viajes_display = viajes_activos[[
                'producto', 'campesino', 'ciudad', 'cantidad_kg', 
                'progreso_viaje', 'distancia_restante_km'
            ]].copy()
            
            viajes_display['progreso_viaje'] = viajes_display['progreso_viaje'].apply(lambda x: f"{x*100:.1f}%")
            viajes_display['distancia_restante_km'] = viajes_display['distancia_restante_km'].apply(lambda x: f"{x:.1f} km" if pd.notna(x) else "N/A")
            viajes_display.columns = ['Producto', 'Campesino', 'Ciudad', 'Cantidad (kg)', 'Progreso', 'Distancia Restante']
            
            st.dataframe(viajes_display, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Estadísticas de productos en venta
        if not productos_recogidos.empty:
            st.markdown("### 🛒 Productos Disponibles para Venta")
            
            col_v1, col_v2, col_v3 = st.columns(3)
            
            with col_v1:
                total_kg = productos_recogidos['cantidad_kg'].sum()
                st.metric("⚖️ Total kg en Inventario", f"{total_kg:.0f}")
            
            with col_v2:
                valor_inventario = productos_recogidos['precio_predicho'].fillna(productos_recogidos['precio']).sum()
                st.metric("💰 Valor Total Inventario", f"${valor_inventario:,.0f}")
            
            with col_v3:
                productos_unicos = productos_recogidos['producto'].nunique()
                st.metric("🌾 Tipos de Productos", productos_unicos)
            
            # Gráfico de productos
            st.markdown("#### 📊 Distribución por Producto")
            
            import plotly.express as px
            
            productos_count = productos_recogidos.groupby('producto').agg({
                'cantidad_kg': 'sum',
                'precio': 'sum'
            }).reset_index()
            
            fig = px.pie(
                productos_count,
                values='cantidad_kg',
                names='producto',
                title='Distribución de Inventario por Producto (kg)',
                hole=0.4
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Resumen total
        st.markdown("### 💼 Resumen Total de Operaciones")
        
        todos_productos = df_notif[df_notif['transportista_asignado'] == nombre_transportista]
        
        if not todos_productos.empty:
            col_r1, col_r2, col_r3, col_r4 = st.columns(4)
            
            with col_r1:
                total_operaciones = len(todos_productos)
                st.metric("📦 Total Operaciones", total_operaciones)
            
            with col_r2:
                total_kg_transportado = todos_productos['cantidad_kg'].sum()
                st.metric("⚖️ Total kg Transportados", f"{total_kg_transportado:.0f}")
            
            with col_r3:
                tasa_conversion = (len(productos_vendidos) / len(productos_recogidos) * 100) if len(productos_recogidos) > 0 else 0
                st.metric("📈 Tasa de Conversión", f"{tasa_conversion:.1f}%")
            
            with col_r4:
                ciudades_atendidas = todos_productos['ciudad'].nunique()
                st.metric("🏙️ Ciudades Atendidas", ciudades_atendidas)


# MAIN
if __name__ == '__main__':
    st.set_page_config(page_title="AgroTransport Pro - Transportista", page_icon="🚚", layout="wide")
    view_transportista()