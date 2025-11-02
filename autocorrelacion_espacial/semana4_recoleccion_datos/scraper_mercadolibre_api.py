#!/usr/bin/env python3
"""
Scraper de Mercado Libre Chile - API Oficial
Categoría: Inmuebles (MLC1459)
100% Legal y sin bloqueos
"""

import requests
import pandas as pd
import time
from datetime import datetime
import os
import json

class MercadoLibreScraper:
    """Scraper para propiedades de Mercado Libre Chile"""
    
    BASE_URL = "https://api.mercadolibre.com"
    SITE = "MLC"  # Chile
    CATEGORY_INMUEBLES = "MLC1459"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Educational Project)'
        })
    
    def buscar_propiedades(self, comuna=None, tipo=None, limit=1000):
        """
        Busca propiedades en Mercado Libre
        
        Args:
            comuna: Comuna específica (ej: "Ñuñoa", "La Reina")
            tipo: Tipo de propiedad ("Departamento", "Casa")
            limit: Número máximo de resultados
        """
        
        print(f"\n🔍 Buscando propiedades en Mercado Libre...")
        if comuna:
            print(f"   📍 Comuna: {comuna}")
        if tipo:
            print(f"   🏠 Tipo: {tipo}")
        
        propiedades = []
        offset = 0
        max_per_request = 50  # Límite de ML
        
        while offset < limit:
            # Construir query
            params = {
                'category': self.CATEGORY_INMUEBLES,
                'limit': min(max_per_request, limit - offset),
                'offset': offset
            }
            
            # Filtros
            if comuna:
                params['q'] = comuna
            
            try:
                # Request
                url = f"{self.BASE_URL}/sites/{self.SITE}/search"
                response = self.session.get(url, params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                results = data.get('results', [])
                
                if not results:
                    print(f"   ℹ️  No hay más resultados (offset={offset})")
                    break
                
                print(f"   ✓ Obtenidos {len(results)} resultados (offset={offset})")
                
                # Procesar cada propiedad
                for item in results:
                    prop = self._extraer_info_basica(item)
                    
                    # Obtener detalles adicionales
                    detalles = self._obtener_detalles(item['id'])
                    if detalles:
                        prop.update(detalles)
                    
                    propiedades.append(prop)
                    
                    # Rate limiting
                    time.sleep(0.1)
                
                offset += len(results)
                
                # Verificar si hay más páginas
                if offset >= data.get('paging', {}).get('total', 0):
                    break
                
                # Pausa entre páginas
                time.sleep(1)
                
            except requests.RequestException as e:
                print(f"   ⚠️  Error en request: {e}")
                break
        
        print(f"\n✅ Total propiedades obtenidas: {len(propiedades)}")
        return propiedades
    
    def _extraer_info_basica(self, item):
        """Extrae información básica del listado"""
        
        # Precio
        precio = item.get('price', None)
        moneda = item.get('currency_id', 'CLP')
        
        # Ubicación
        location = item.get('location', {})
        address = location.get('address_line', '')
        city = location.get('city', {}).get('name', '')
        state = location.get('state', {}).get('name', '')
        
        # Coordenadas
        lat = location.get('latitude', None)
        lon = location.get('longitude', None)
        
        return {
            'id': item.get('id'),
            'titulo': item.get('title'),
            'precio': precio,
            'moneda': moneda,
            'direccion': address,
            'ciudad': city,
            'region': state,
            'latitud': lat,
            'longitud': lon,
            'permalink': item.get('permalink'),
            'thumbnail': item.get('thumbnail'),
            'fecha_obtencion': datetime.now().isoformat()
        }
    
    def _obtener_detalles(self, item_id):
        """Obtiene detalles completos de una propiedad"""
        
        try:
            url = f"{self.BASE_URL}/items/{item_id}"
            response = self.session.get(url, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            
            # Extraer atributos
            attributes = {}
            for attr in data.get('attributes', []):
                attr_id = attr.get('id')
                attr_value = attr.get('value_name') or attr.get('value_struct', {}).get('number')
                
                if attr_id and attr_value:
                    attributes[attr_id] = attr_value
            
            # Mapear atributos comunes
            return {
                'superficie_total': self._extraer_numero(attributes.get('TOTAL_AREA')),
                'superficie_cubierta': self._extraer_numero(attributes.get('COVERED_AREA')),
                'dormitorios': self._extraer_numero(attributes.get('BEDROOMS')),
                'banos': self._extraer_numero(attributes.get('BATHROOMS')),
                'estacionamientos': self._extraer_numero(attributes.get('PARKING_LOT_SIZE')),
                'antiguedad': attributes.get('PROPERTY_AGE'),
                'tipo_propiedad': attributes.get('PROPERTY_TYPE'),
                'operacion': attributes.get('OPERATION_TYPE'),
                'condicion': data.get('condition'),
                'descripcion': data.get('description', '')[:500]  # Primeros 500 chars
            }
            
        except Exception as e:
            print(f"   ⚠️  Error obteniendo detalles de {item_id}: {e}")
            return {}
    
    def _extraer_numero(self, valor):
        """Extrae número de string"""
        if valor is None:
            return None
        try:
            return float(str(valor).replace(',', ''))
        except:
            return None
    
    def guardar_resultados(self, propiedades, output_dir='datos_procesados'):
        """Guarda resultados en múltiples formatos"""
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # DataFrame
        df = pd.DataFrame(propiedades)
        
        # CSV
        csv_file = f'{output_dir}/mercadolibre_{timestamp}.csv'
        df.to_csv(csv_file, index=False)
        print(f"💾 CSV: {csv_file}")
        
        # JSON
        json_file = f'{output_dir}/mercadolibre_{timestamp}.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(propiedades, f, ensure_ascii=False, indent=2)
        print(f"💾 JSON: {json_file}")
        
        # Estadísticas
        print(f"\n📊 ESTADÍSTICAS:")
        print(f"   Total propiedades: {len(df)}")
        
        if 'precio' in df.columns:
            precios = df['precio'].dropna()
            print(f"   Precio promedio: {precios.mean():,.0f} {df['moneda'].mode()[0]}")
            print(f"   Precio mediana: {precios.median():,.0f}")
        
        if 'tipo_propiedad' in df.columns:
            print(f"\n   Tipos:")
            for tipo, count in df['tipo_propiedad'].value_counts().head(3).items():
                print(f"      • {tipo}: {count}")
        
        if 'ciudad' in df.columns:
            print(f"\n   Comunas:")
            for ciudad, count in df['ciudad'].value_counts().head(5).items():
                print(f"      • {ciudad}: {count}")
        
        return df


def main():
    """Función principal"""
    
    print("=" * 80)
    print("🏠 SCRAPER MERCADO LIBRE CHILE - API OFICIAL")
    print("=" * 80)
    
    scraper = MercadoLibreScraper()
    
    # Comunas de interés
    comunas = ['Ñuñoa', 'La Reina', 'Santiago', 'Estación Central']
    
    todas_propiedades = []
    
    for comuna in comunas:
        props = scraper.buscar_propiedades(comuna=comuna, limit=250)
        todas_propiedades.extend(props)
        time.sleep(2)  # Pausa entre comunas
    
    # Remover duplicados
    df_temp = pd.DataFrame(todas_propiedades)
    df_temp = df_temp.drop_duplicates(subset=['id'])
    propiedades_unicas = df_temp.to_dict('records')
    
    print(f"\n📋 Total propiedades únicas: {len(propiedades_unicas)}")
    
    # Guardar
    df_final = scraper.guardar_resultados(propiedades_unicas)
    
    print("\n✅ SCRAPING COMPLETADO")
    print("\n📍 Siguiente paso: geocodificar y enriquecer con características espaciales")
    
    return df_final


if __name__ == "__main__":
    df = main()
