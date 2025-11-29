import requests
import json
import time
import random
import csv
import os
from datetime import datetime
from bs4 import BeautifulSoup

# Headers para simular un navegador real
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'es-CL,es;q=0.9,en;q=0.8',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}

# URLs a scrapear (todas las comunas y tipos de propiedad)
URLS_TO_SCRAPE = {
    1: {
        'nombre': 'Departamentos - Estación Central',
        'url': 'https://www.portalinmobiliario.com/venta/departamento/estacion-central-metropolitana',
        'comuna': 'Estación Central',
        'tipo': 'departamento'
    },
    2: {
        'nombre': 'Casas - Estación Central',
        'url': 'https://www.portalinmobiliario.com/venta/casa/estacion-central-metropolitana',
        'comuna': 'Estación Central',
        'tipo': 'casa'
    },
    3: {
        'nombre': 'Departamentos - Santiago',
        'url': 'https://www.portalinmobiliario.com/venta/departamento/santiago-metropolitana',
        'comuna': 'Santiago',
        'tipo': 'departamento'
    },
    4: {
        'nombre': 'Casas - Santiago',
        'url': 'https://www.portalinmobiliario.com/venta/casa/santiago-metropolitana',
        'comuna': 'Santiago',
        'tipo': 'casa'
    },
    5: {
        'nombre': 'Departamentos - Ñuñoa',
        'url': 'https://www.portalinmobiliario.com/venta/departamento/nunoa-metropolitana',
        'comuna': 'Ñuñoa',
        'tipo': 'departamento'
    },
    6: {
        'nombre': 'Casas - Ñuñoa',
        'url': 'https://www.portalinmobiliario.com/venta/casa/nunoa-metropolitana',
        'comuna': 'Ñuñoa',
        'tipo': 'casa'
    },
    7: {
        'nombre': 'Departamentos - La Reina',
        'url': 'https://www.portalinmobiliario.com/venta/departamento/la-reina-metropolitana',
        'comuna': 'La Reina',
        'tipo': 'departamento'
    },
    8: {
        'nombre': 'Casas - La Reina',
        'url': 'https://www.portalinmobiliario.com/venta/casa/la-reina-metropolitana',
        'comuna': 'La Reina',
        'tipo': 'casa'
    },
}

# Configuración de delays y paginación
DELAY_MIN = 5  # Segundos mínimos entre peticiones
DELAY_MAX = 60  # Segundos máximos entre peticiones
ITEMS_POR_PAGINA = 48  # Cantidad de items por página (según el sitio)


def inicializar_csv(archivo):
    """Inicializa el archivo CSV con los headers."""
    headers = [
        'fecha_extraccion',
        'comuna',
        'tipo_propiedad',
        'titulo',
        'precio',
        'moneda',
        'dormitorios',
        'banos',
        'metros_utiles',
        'caracteristicas_raw',
        'ubicacion',
        'url'
    ]
    with open(archivo, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(headers)
    print(f"📁 Archivo CSV creado: {archivo}")


def guardar_propiedades_csv(archivo, propiedades, comuna, tipo_propiedad):
    """Guarda una lista de propiedades en el archivo CSV."""
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(archivo, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f, delimiter=';')
        
        for prop in propiedades:
            datos = extraer_datos_propiedad(prop)
            
            # Parsear características para extraer dormitorios, baños y metros
            dormitorios = ""
            banos = ""
            metros = ""
            
            for caract in datos['caracteristicas']:
                caract_lower = caract.lower()
                if 'dormitorio' in caract_lower:
                    dormitorios = caract
                elif 'baño' in caract_lower:
                    banos = caract
                elif 'm²' in caract_lower or 'útiles' in caract_lower:
                    metros = caract
            
            row = [
                fecha,
                comuna,
                tipo_propiedad,
                datos['titulo'],
                datos['precio'] if datos['precio'] else "",
                datos['moneda'],
                dormitorios,
                banos,
                metros,
                ' | '.join(datos['caracteristicas']) if datos['caracteristicas'] else "",
                datos['ubicacion'],
                datos['url'] if datos['url'] else ""
            ]
            writer.writerow(row)
    
    print(f"   💾 Guardadas {len(propiedades)} propiedades en CSV")


def esperar_aleatorio():
    """Espera un tiempo aleatorio entre peticiones para simular comportamiento humano."""
    delay = random.uniform(DELAY_MIN, DELAY_MAX)
    print(f"⏳ Esperando {delay:.1f} segundos antes de la siguiente petición...")
    time.sleep(delay)


def scrape_page(url):
    """Función para scrapear una sola página y devolver sus resultados."""
    print(f"\n🔍 Scrapeando URL: {url}")
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        script_tag = soup.find('script', {'id': '__PRELOADED_STATE__'})
        
        if not script_tag:
            print("❌ No se encontró la etiqueta __PRELOADED_STATE__ en esta página.")
            return [], None, None

        json_data = json.loads(script_tag.string)
        
        initial_state = json_data.get('pageState', {}).get('initialState', {})
        total_results = initial_state.get('melidata_track', {}).get('event_data', {}).get('total', 0)
        items_per_page = initial_state.get('melidata_track', {}).get('event_data', {}).get('limit', 50)
        
        results = initial_state.get('results', [])
        print(f"✅ Encontrados {len(results)} resultados en esta página")
        return results, total_results, items_per_page

    except requests.exceptions.RequestException as e:
        print(f"❌ Error al obtener la URL {url}: {e}")
        return [], None, None
    except Exception as e:
        print(f"❌ Ocurrió un error procesando la página {url}: {e}")
        return [], None, None


def extraer_datos_propiedad(item):
    """Extrae los datos estructurados de una propiedad."""
    polycard = item.get('polycard', {})
    components = polycard.get('components', [])
    
    datos = {
        'titulo': "No disponible",
        'precio': None,
        'moneda': "",
        'caracteristicas': [],
        'ubicacion': "No disponible",
        'url': None,
    }
    
    # Extraer URL del track
    track = polycard.get('tracks', {}).get('melidata_track', {})
    datos['url'] = track.get('event_data', {}).get('url')

    for component in components:
        tipo = component.get('type')
        if tipo == 'title':
            datos['titulo'] = component.get('title', {}).get('text')
        elif tipo == 'price':
            price_info = component.get('price', {})
            datos['precio'] = price_info.get('current_price', {}).get('value')
            datos['moneda'] = price_info.get('current_price', {}).get('currency', '')
        elif tipo == 'attributes_list':
            datos['caracteristicas'] = component.get('attributes_list', {}).get('texts', [])
        elif tipo == 'location':
            datos['ubicacion'] = component.get('location', {}).get('text')
    
    return datos


def mostrar_propiedad(datos):
    """Muestra los datos de una propiedad de forma legible."""
    print(f"  📍 Título: {datos['titulo']}")
    precio_str = f"{datos['precio']:,.0f} {datos['moneda']}" if datos['precio'] else "No disponible"
    print(f"  💰 Precio: {precio_str}")
    if datos['caracteristicas']:
        print(f"  🏠 Características: {', '.join(datos['caracteristicas'])}")
    print(f"  📌 Ubicación: {datos['ubicacion']}")
    if datos['url']:
        print(f"  🔗 URL: {datos['url']}")
    print("-" * 50)


def mostrar_menu():
    """Muestra el menú de selección."""
    print("\n" + "=" * 60)
    print("🏠 SCRAPER DE PORTAL INMOBILIARIO")
    print("=" * 60)
    print("\n📋 Seleccione qué desea scrapear:\n")
    
    for key, value in URLS_TO_SCRAPE.items():
        print(f"   [{key}] {value['nombre']}")
    
    print(f"\n   [9] 🚀 TODAS las opciones")
    print(f"\n   [C] 🔄 CONTINUAR scraping anterior")
    print(f"   [0] ❌ Salir")
    print("\n" + "-" * 60)


def obtener_seleccion():
    """Obtiene la selección del usuario."""
    while True:
        mostrar_menu()
        try:
            opcion = input("\n👉 Ingrese su opción (puede ingresar varias separadas por coma, ej: 1,3,5): ").strip().upper()
            
            if opcion == '0':
                return None, None
            
            if opcion == 'C':
                return 'CONTINUAR', None
            
            if opcion == '9':
                return list(URLS_TO_SCRAPE.keys()), None
            
            # Parsear múltiples opciones
            opciones = [int(x.strip()) for x in opcion.split(',')]
            
            # Validar que todas las opciones sean válidas
            opciones_validas = []
            for op in opciones:
                if op in URLS_TO_SCRAPE:
                    opciones_validas.append(op)
                else:
                    print(f"⚠️  Opción {op} no válida, ignorando...")
            
            if opciones_validas:
                return opciones_validas, None
            else:
                print("❌ Ninguna opción válida ingresada. Intente de nuevo.")
                
        except ValueError:
            print("❌ Entrada inválida. Por favor ingrese números separados por coma.")


def obtener_parametros_continuacion():
    """Obtiene los parámetros para continuar un scraping anterior."""
    print("\n" + "=" * 60)
    print("🔄 CONTINUAR SCRAPING ANTERIOR")
    print("=" * 60)
    
    # Mostrar opciones disponibles
    print("\n📋 ¿Qué búsqueda desea continuar?\n")
    for key, value in URLS_TO_SCRAPE.items():
        print(f"   [{key}] {value['nombre']}")
    
    try:
        opcion = int(input("\n👉 Seleccione la opción: ").strip())
        if opcion not in URLS_TO_SCRAPE:
            print("❌ Opción no válida.")
            return None
        
        # Pedir el offset desde donde continuar
        print(f"\n📍 Último offset exitoso (ej: si quedó en _Desde_337, ingrese 337)")
        ultimo_offset = int(input("👉 Ingrese el último offset exitoso: ").strip())
        
        # Pedir archivo existente o crear nuevo
        archivo_existente = input("\n📁 ¿Desea agregar a un archivo existente? (ingrese nombre o Enter para nuevo): ").strip()
        
        if archivo_existente:
            if not os.path.exists(archivo_existente):
                print(f"⚠️  Archivo {archivo_existente} no existe. Se creará uno nuevo.")
                archivo_existente = None
        
        return {
            'opcion': opcion,
            'offset_inicial': ultimo_offset + ITEMS_POR_PAGINA,
            'archivo': archivo_existente
        }
        
    except ValueError:
        print("❌ Entrada inválida.")
        return None


def scrapear_continuacion(params):
    """Continúa el scraping desde donde quedó."""
    opcion = params['opcion']
    offset_inicial = params['offset_inicial']
    archivo_existente = params.get('archivo')
    
    info = URLS_TO_SCRAPE[opcion]
    base_url = info['url']
    comuna = info['comuna']
    tipo_propiedad = info['tipo']
    
    # Determinar archivo de salida
    if archivo_existente:
        archivo_salida = archivo_existente
        print(f"\n📁 Continuando en archivo: {archivo_salida}")
    else:
        fecha_actual = datetime.now().strftime("%Y%m%d_%H%M%S")
        archivo_salida = f"propiedades_{fecha_actual}.csv"
        inicializar_csv(archivo_salida)
    
    print(f"\n🔄 Continuando scraping de: {info['nombre']}")
    print(f"📍 Empezando desde offset: {offset_inicial}")
    print(f"⏱️  Delay entre peticiones: {DELAY_MIN}-{DELAY_MAX} segundos")
    
    total_propiedades = 0
    paginas_scrapeadas = 0
    offset = offset_inicial
    
    # Obtener total de resultados para calcular páginas restantes
    print(f"\n{'='*60}")
    print(f"📊 Obteniendo información de paginación...")
    results_inicial, total_results, _ = scrape_page(base_url)
    
    if total_results:
        total_paginas = (total_results // ITEMS_POR_PAGINA) + 1
        pagina_actual = offset_inicial // ITEMS_POR_PAGINA
        max_paginas = min(total_paginas, 2000 // ITEMS_POR_PAGINA)
        paginas_restantes = max_paginas - pagina_actual
        print(f"📈 Total en sitio: {total_results} | Página actual: {pagina_actual}/{max_paginas}")
        print(f"📄 Páginas restantes: {paginas_restantes}")
    else:
        max_paginas = 42  # Estimado por defecto
        paginas_restantes = max_paginas
    
    esperar_aleatorio()
    
    # Continuar paginación
    while True:
        paginated_url = f"{base_url}/_Desde_{offset}_NoIndex_True"
        page_results, _, _ = scrape_page(paginated_url)
        
        if not page_results:
            print("🛑 No hay más resultados.")
            break
        
        propiedades = [item for item in page_results if item.get('id') == 'POLYCARD']
        
        # Guardar en CSV (modo append si es archivo existente)
        if archivo_existente:
            guardar_propiedades_csv(archivo_salida, propiedades, comuna, tipo_propiedad)
        else:
            guardar_propiedades_csv(archivo_salida, propiedades, comuna, tipo_propiedad)
        
        total_propiedades += len(propiedades)
        paginas_scrapeadas += 1
        
        pagina_actual = offset // ITEMS_POR_PAGINA
        print(f"   📊 Página {pagina_actual}/{max_paginas} | Offset: {offset} | Total nuevas: {total_propiedades}")
        
        offset += ITEMS_POR_PAGINA
        
        # Verificar si llegamos al límite
        if offset > 2000:
            print("🛑 Alcanzado límite de 2000 resultados del sitio.")
            break
        
        esperar_aleatorio()
    
    # Resumen
    print("\n" + "=" * 60)
    print("📋 RESUMEN CONTINUACIÓN")
    print("=" * 60)
    print(f"\n✅ Páginas scrapeadas en esta sesión: {paginas_scrapeadas}")
    print(f"✅ Propiedades nuevas extraídas: {total_propiedades}")
    print(f"📁 Archivo: {archivo_salida}")
    print(f"📍 Ubicación: {os.path.abspath(archivo_salida)}")
    
    return archivo_salida


def scrapear_seleccion(opciones_seleccionadas):
    """Ejecuta el scraping para las opciones seleccionadas."""
    # Generar nombre de archivo
    fecha_actual = datetime.now().strftime("%Y%m%d_%H%M%S")
    archivo_salida = f"propiedades_{fecha_actual}.csv"
    
    # Inicializar archivo CSV
    inicializar_csv(archivo_salida)
    
    total_propiedades = 0
    total_urls = len(opciones_seleccionadas)
    
    print(f"\n🚀 Iniciando scraping de {total_urls} búsqueda(s)...")
    print(f"📁 Guardando en: {archivo_salida}")
    print(f"⏱️  Delay entre peticiones: {DELAY_MIN}-{DELAY_MAX} segundos")
    
    for i, opcion in enumerate(opciones_seleccionadas):
        info = URLS_TO_SCRAPE[opcion]
        base_url = info['url']
        comuna = info['comuna']
        tipo_propiedad = info['tipo']
        
        print(f"\n{'='*60}")
        print(f"📊 Procesando {i+1}/{total_urls}: {info['nombre']}")
        print(f"{'='*60}")
        
        # Primera página
        results, total_results, _ = scrape_page(base_url)
        paginas_scrapeadas = 1
        
        if results:
            propiedades = [item for item in results if item.get('id') == 'POLYCARD']
            guardar_propiedades_csv(archivo_salida, propiedades, comuna, tipo_propiedad)
            total_propiedades += len(propiedades)
            
            if total_results:
                print(f"📈 Total en el sitio: {total_results} propiedades")
                total_paginas = (total_results // ITEMS_POR_PAGINA) + 1
                # Límite real del sitio es ~2000 resultados
                max_paginas = min(total_paginas, 2000 // ITEMS_POR_PAGINA)
                print(f"📄 Páginas a scrapear: {max_paginas}")
            else:
                max_paginas = 1
            
            # Paginación
            offset = ITEMS_POR_PAGINA + 1
            
            while paginas_scrapeadas < max_paginas:
                esperar_aleatorio()
                
                paginated_url = f"{base_url}/_Desde_{offset}_NoIndex_True"
                page_results, _, _ = scrape_page(paginated_url)
                
                if not page_results:
                    print("🛑 No hay más resultados.")
                    break
                
                propiedades = [item for item in page_results if item.get('id') == 'POLYCARD']
                guardar_propiedades_csv(archivo_salida, propiedades, comuna, tipo_propiedad)
                total_propiedades += len(propiedades)
                
                paginas_scrapeadas += 1
                offset += ITEMS_POR_PAGINA
                
                print(f"   📊 Progreso: {paginas_scrapeadas}/{max_paginas} páginas | Total: {total_propiedades} propiedades")
        
        print(f"\n✅ Completado: {paginas_scrapeadas} páginas para {info['nombre']}")
        
        # Esperar antes de la siguiente URL
        if i < total_urls - 1:
            esperar_aleatorio()
    
    # Resumen final
    print("\n" + "=" * 60)
    print("📋 RESUMEN FINAL")
    print("=" * 60)
    print(f"\n✅ Total de propiedades extraídas: {total_propiedades}")
    print(f"📁 Archivo guardado: {archivo_salida}")
    print(f"📍 Ubicación: {os.path.abspath(archivo_salida)}")
    
    return archivo_salida


# --- Script Principal ---
if __name__ == "__main__":
    try:
        opciones, _ = obtener_seleccion()
        
        if opciones is None:
            print("\n👋 ¡Hasta luego!")
        
        elif opciones == 'CONTINUAR':
            # Modo continuación
            params = obtener_parametros_continuacion()
            if params:
                confirmar = input("\n¿Continuar scraping? (s/n): ").strip().lower()
                if confirmar == 's':
                    scrapear_continuacion(params)
                else:
                    print("\n❌ Operación cancelada.")
        
        else:
            # Modo normal
            print("\n📝 Has seleccionado:")
            for op in opciones:
                print(f"   ✓ {URLS_TO_SCRAPE[op]['nombre']}")
            
            confirmar = input("\n¿Continuar? (s/n): ").strip().lower()
            
            if confirmar == 's':
                scrapear_seleccion(opciones)
            else:
                print("\n❌ Operación cancelada.")
                
    except KeyboardInterrupt:
        print("\n\n⚠️  Scraping interrumpido por el usuario.")
        print("💾 Los datos obtenidos hasta el momento fueron guardados.")
        print("💡 Tip: Use la opción [C] para continuar desde donde quedó.")