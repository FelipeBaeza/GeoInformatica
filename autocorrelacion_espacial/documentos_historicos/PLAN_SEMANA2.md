# SEMANA 2: INGENIERÍA DE CARACTERÍSTICAS ESPACIALES

## Objetivo
Generar características espaciales cuantitativas que alimentarán el sistema de recomendaciones personalizadas, calculando distancias, densidades y índices de accesibilidad para cada ubicación en Santiago.

## Arquitectura de Características Espaciales

### 1. CARACTERÍSTICAS DE DISTANCIA
- **Distancia al metro más cercano**: Fundamental para movilidad urbana
- **Distancia a colegios por nivel**: Básica, media, superior
- **Distancia a servicios de salud**: Hospitales, consultorios, farmacias
- **Distancia a servicios públicos**: Municipalidades, comisarías, bomberos
- **Distancia a áreas verdes**: Parques, plazas, espacios recreativos
- **Distancia a comercio**: Centros comerciales, tiendas, servicios

### 2. CARACTERÍSTICAS DE DENSIDAD (Radios: 300m, 600m, 1km)
- **Densidad educacional**: N° establecimientos por km²
- **Densidad de salud**: Centros médicos por km²
- **Densidad comercial**: Tiendas y servicios por km²
- **Densidad de transporte**: Estaciones y paradas por km²
- **Densidad de seguridad**: Comisarías y cuarteles por km²
- **Densidad recreativa**: Espacios verdes y ocio por km²

### 3. ÍNDICES DE ACCESIBILIDAD COMPUESTA
- **Índice de Accesibilidad Educativa**: Combina distancia y densidad
- **Índice de Accesibilidad a Salud**: Pondera por tipo de servicio
- **Índice de Conectividad**: Basado en transporte público
- **Índice de Calidad del Entorno**: Espacios verdes vs contaminación
- **Índice de Seguridad Percibida**: Basado en infraestructura de seguridad

### 4. GRILLA DE EVALUACIÓN
- **Grid regular de 200x200m** cubriendo área metropolitana
- **~15,000 puntos** de evaluación para Santiago
- **Características calculadas en cada punto** de la grilla
- **Interpolación espacial** para ubicaciones específicas

## Metodología de Cálculo

### Distancias Euclidianas
Utilizaremos la distancia euclidiana (línea recta) como aproximación inicial, calculada eficientemente usando el sistema UTM 19S que permite medidas precisas en metros.

### Buffers Circulares
Para las densidades, crearemos buffers circulares de 300m, 600m y 1km alrededor de cada punto de la grilla, contando las amenidades dentro de cada radio.

### Índices Normalizados
Todos los índices se normalizarán a escala 0-10 para facilitar la interpretación y comparación entre diferentes tipos de características.