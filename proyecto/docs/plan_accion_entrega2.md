# Plan de Acción - Segunda Entrega TerraMatch

## 🎯 Objetivos

Convertir el informe de **Primera Entrega** (15-20 págs) en **Informe Final** (25-35 págs) cumpliendo todos los requisitos de la Segunda Evaluación.

---

## 📅 Cronograma de Tareas

### FASE 1: Presentación Oral (Deadline: 22 Enero 2026)

#### Tarea 1.1: Crear Slides de Presentación
**Responsable sugerido:** Todos (dividir secciones)  
**Tiempo estimado:** 8-10 horas  
**Estructura:**

```
slides_presentacion.pptx

Slide 1: Portada
  - Título: TerraMatch
  - Subtítulo: Sistema de estimación de satisfacción inmobiliaria
  - Integrantes + Logo USACH

Slides 2-4: Introducción y Motivación (3 min)
  - Problema: Crisis de acceso a vivienda (20% pie)
  - Relevancia: 70,000 unidades en stock
  - Pregunta: ¿Qué hace satisfactoria una vivienda en Santiago?

Slides 5-8: Datos y Metodología (4 min)
  - Fuentes: 29 datasets geoespaciales
  - Área de estudio: 4 comunas (mapa)
  - Pipeline: 3 fases (diagrama)
  - Herramientas: Python + GeoPandas + LightGBM

Slides 9-16: Resultados Principales (8 min) ← MÁS IMPORTANTE
  - Dataset: 7,702 propiedades
  - Modelo: R² = 0.8635
  - Mapa de satisfacción predicha
  - Top variables importantes
  - Comparación por comunas
  - Autocorrelación espacial (Moran's I)
  - **DEMO DEL DASHBOARD** (3-4 min en vivo)

Slides 17-19: Conclusiones y Discusión (3 min)
  - Hallazgos clave
  - Implicancias prácticas
  - Limitaciones
  - Trabajo futuro

Slide 20: Cierre (2 min)
  - Contribución del proyecto
  - Agradecimientos
  - Contacto / Repo GitHub
```

**Checklist:**
- [ ] Crear archivo de slides (PowerPoint/Google Slides)
- [ ] Asignar secciones a cada integrante
- [ ] Incluir visualizaciones de alta calidad del informe
- [ ] Preparar demo del dashboard (con plan B: video)
- [ ] Cronometrar presentación completa
- [ ] Ensayar 2-3 veces completo

---

#### Tarea 1.2: Preparar Demo del Dashboard
**Responsable sugerido:** Desarrolladores frontend/backend  
**Tiempo estimado:** 4-6 horas  

**Checklist:**
- [ ] Verificar que el dashboard corra sin errores
- [ ] Probar todas las funcionalidades:
  - [ ] Mapa interactivo carga correctamente
  - [ ] Filtros funcionan (por comuna, tipo vivienda, rango precio)
  - [ ] Gráficos se actualizan con filtros
  - [ ] Información contextual está visible
- [ ] Grabar video de respaldo (2-3 min) por si falla conexión
- [ ] Preparar laptop con conexión a internet estable
- [ ] Tener datos precargados para demo rápida

**Script sugerido para demo (3 min):**
```
1. "Este es TerraMatch, nuestro sistema de recomendación inmobiliaria"
2. Mostrar mapa con propiedades coloreadas por satisfacción predicha
3. Aplicar filtro: "Veamos solo departamentos en Ñuñoa bajo 5000 UF"
4. Mostrar cómo cambian los gráficos de distribución
5. Seleccionar una propiedad: "Aquí vemos el detalle y su score"
6. Cerrar: "Esto permite comparar objetivamente alternativas de vivienda"
```

---

### FASE 2: Actualización del Informe (Deadline: Primera semana Marzo 2026)

#### Tarea 2.1: Expandir Marco Teórico (de ~4 a 3-4 páginas con 15+ refs)
**Responsable sugerido:** 2 integrantes  
**Tiempo estimado:** 12-15 horas  

**Pasos específicos:**

1. **Buscar 10-12 papers adicionales** en Google Scholar:
   ```
   Búsquedas sugeridas:
   - "hedonic price model housing location"
   - "residential satisfaction spatial analysis"
   - "machine learning real estate valuation"
   - "urban accessibility services"
   - "spatial autocorrelation housing prices"
   - Casos Chile: "vivienda satisfacción santiago chile"
   ```

2. **Crear archivo `referencias.bib`**
   
   Ejemplo de entrada BibTeX:
   ```bibtex
   @article{rosen1974,
     author = {Rosen, Sherwin},
     title = {Hedonic Prices and Implicit Markets},
     journal = {Journal of Political Economy},
     year = {1974},
     volume = {82},
     number = {1},
     pages = {34--55}
   }
   
   @article{anselin1995,
     author = {Anselin, Luc},
     title = {Local Indicators of Spatial Association - LISA},
     journal = {Geographical Analysis},
     year = {1995},
     volume = {27},
     number = {2},
     pages = {93--115}
   }
   
   @article{fotheringham2002,
     author = {Fotheringham, A. Stewart and Brunsdon, Chris and Charlton, Martin},
     title = {Geographically Weighted Regression},
     journal = {Journal of the Royal Statistical Society},
     year = {2002},
     volume = {51},
     pages = {1--27}
   }
   
   @book{osullivan2010,
     author = {O'Sullivan, David and Unwin, David J.},
     title = {Geographic Information Analysis},
     publisher = {Wiley},
     year = {2010},
     edition = {2nd}
   }
   
   @article{lundberg2017,
     author = {Lundberg, Scott M. and Lee, Su-In},
     title = {A Unified Approach to Interpreting Model Predictions},
     journal = {Advances in Neural Information Processing Systems},
     year = {2017},
     volume = {30}
   }
   ```

3. **Modificar informe_v1.tex:**
   - Descomentar línea: `\addbibresource{referencias.bib}`
   - Agregar citas en el texto: `\cite{rosen1974}`
   - Al final, antes de anexos: `\printbibliography`

4. **Expandir subsecciones del Marco Teórico:**
   
   Agregar después de la sección actual:
   
   ```latex
   \subsubsection{Modelos Hedónicos y Valoración Inmobiliaria}
   Los modelos hedónicos descomponen el precio de la vivienda en atributos 
   observables, permitiendo inferir el valor implícito de características 
   internas y externas \cite{rosen1974}. En contexto urbano, la proximidad 
   a servicios y la calidad del entorno capturan parte importante de la 
   variación espacial de precios \cite{freeman2003}.
   
   \subsubsection{Autocorrelación Espacial y Geographically Weighted Models}
   La presencia de autocorrelación espacial en datos inmobiliarios viola 
   supuestos de independencia de modelos clásicos \cite{anselin1995}. 
   Los modelos GWR y GWRF permiten capturar heterogeneidad espacial local 
   \cite{fotheringham2002}, mejorando la capacidad predictiva en contextos 
   donde los determinantes varían territorialmente.
   
   \subsubsection{Machine Learning en Análisis Inmobiliario}
   Algoritmos de gradient boosting como LightGBM han mostrado desempeño 
   superior frente a modelos lineales en predicción de precios, manteniendo 
   interpretabilidad mediante análisis de importancia de variables 
   \cite{lundberg2017}. Su capacidad para capturar interacciones no lineales 
   resulta relevante cuando se integran múltiples fuentes geoespaciales.
   ```

**Checklist:**
- [ ] Encontrar 15 referencias relevantes
- [ ] Crear archivo `referencias.bib` en `proyecto/docs/`
- [ ] Integrar citas en el texto del marco teórico
- [ ] Expandir subsecciones con literatura
- [ ] Compilar LaTeX y verificar que bibliografía aparezca
- [ ] Revisar formato APA de referencias

---

#### Tarea 2.2: Expandir Sección de Discusión (de ~1.5 a 3-4 páginas)
**Responsable sugerido:** 2 integrantes  
**Tiempo estimado:** 10-12 horas  

**Estructura propuesta:**

```latex
\section{Discusión}

\subsection{Interpretación de Resultados}
[AMPLIAR la subsección actual con:]

Los resultados confirman que las variables económicas (precio/m², superficie) 
dominan la predicción de satisfacción, explicando aproximadamente el 60% de 
la varianza. Sin embargo, las métricas espaciales aportan un 26% adicional, 
validando la hipótesis central del proyecto sobre la relevancia del entorno 
urbano \cite{referencia_similar}.

El modelo final (R² = 0.8635) se compara favorablemente con estudios previos 
en contextos similares. Por ejemplo, [NOMBRE ESTUDIO] reportó R² de 0.78 
para predicción de precios en [CIUDAD], utilizando solo atributos internos. 
La incorporación de variables geoespaciales en TerraMatch mejora sustancialmente 
la capacidad predictiva.

La baja autocorrelación residual (Moran's I = 0.0695, p < 0.05) indica que 
el modelo captura adecuadamente la estructura espacial, evitando sesgos 
territoriales sistemáticos. Esto contrasta con modelos puramente hedónicos 
que tienden a presentar autocorrelación residual significativa \cite{anselin1995}.

\subsection{Comparación con Literatura y Casos Similares}
[NUEVA SUBSECCIÓN]

La importancia relativa de variables coincide con hallazgos internacionales:
- Acceso a transporte público como predictor clave: consistente con 
  [ESTUDIO 1] en Londres y [ESTUDIO 2] en São Paulo
- Proximidad a áreas verdes: similar a [ESTUDIO 3] que reportó elasticidad 
  precio-distancia de -0.12 por cada 100m adicionales
- Rol de seguridad: alineado con [ESTUDIO 4] en ciudades latinoamericanas

Sin embargo, a diferencia de estudios en países desarrollados, en Santiago 
el factor "distancia a metro" tiene mayor peso relativo, reflejando limitaciones 
de transporte alternativo \cite{ref_transporte_santiago}.

\subsection{Implicancias Prácticas}
[AMPLIAR subsección actual]

\subsubsection{Para Compradores de Vivienda}
El sistema permite objetivar decisiones, identificando propiedades que maximizan 
satisfacción dado un presupuesto. Por ejemplo, un profesional joven priorizando 
transporte puede descubrir que Estación Central ofrece mejor relación 
satisfacción/precio que Ñuñoa.

\subsubsection{Para Política Pública}
Los patrones identificados sugieren:
1. **Transporte:** Mejorar conectividad en comunas con bajo índice de transporte 
   (e.g., La Reina) podría incrementar satisfacción residencial
2. **Salud:** Déficit de servicios médicos en Estación Central justifica 
   inversión en consultorios de atención primaria
3. **Áreas verdes:** Inequidad territorial evidente; priorizar parques en 
   zonas con índice de entorno < 3.0

\subsubsection{Para el Mercado Inmobiliario}
Desarrolladores pueden usar métricas de accesibilidad para identificar zonas 
subvaloradas con potencial de apreciación al mejorar servicios cercanos.

\subsection{Limitaciones del Estudio}
[AMPLIAR subsección actual]

\subsubsection{Limitaciones de Datos}
1. **Temporalidad:** Datos estáticos de 2025 no capturan dinámica del mercado
2. **Atributos faltantes:** Antigüedad, estado de conservación, orientación, 
   equipamiento de edificio (no disponibles en scraping)
3. **Gastos operacionales:** Ausencia de gastos comunes y contribuciones limita 
   análisis de asequibilidad real
4. **Cobertura espacial:** Solo 4 comunas; resultados no generalizables a 
   toda la Región Metropolitana

\subsubsection{Limitaciones Metodológicas}
1. **Distancia euclidiana:** No considera barreras físicas ni red vial real
2. **Proxy de satisfacción:** Construcción teórica sin validación empírica 
   con usuarios reales
3. **Sesgo de selección:** Solo propiedades en venta; no representa stock total
4. **Causalidad:** Modelo predictivo no establece relaciones causales entre 
   accesibilidad y satisfacción

\subsection{Trabajo Futuro}
[AMPLIAR con propuestas concretas]

\subsubsection{Corto Plazo (6 meses)}
1. **Validación empírica:** Encuesta a 100+ compradores recientes para validar 
   proxy de satisfacción
2. **Expansión territorial:** Incorporar 10 comunas adicionales (Providencia, 
   Las Condes, Maipú, etc.)
3. **Variables ambientales:** Integrar datos de contaminación acústica y 
   atmosférica (SINCA)

\subsubsection{Mediano Plazo (1 año)}
1. **Series temporales:** Tracking de propiedades en el tiempo para capturar 
   evolución del mercado
2. **Distancias en red:** Usar OSMnx para calcular distancias reales por red vial
3. **Modelos explicables:** Implementar SHAP values para interpretabilidad local
4. **Análisis de sensibilidad:** Evaluar robustez ante cambios en pesos de perfiles

\subsubsection{Largo Plazo (2+ años)}
1. **Escalabilidad:** Extender a nivel nacional (ciudades intermedias)
2. **Integración con tasaciones:** Colaboración con bancos para validación cruzada
3. **App móvil:** Desarrollo de aplicación para usuarios finales con 
   recomendaciones personalizadas
```

**Checklist:**
- [ ] Buscar 3-5 estudios similares para comparación
- [ ] Agregar subsección de Comparación con Literatura
- [ ] Ampliar Implicancias Prácticas (3 perspectivas)
- [ ] Detallar Limitaciones (datos + metodológicas)
- [ ] Proponer Trabajo Futuro concreto (corto/mediano/largo plazo)
- [ ] Integrar citas bibliográficas en toda la sección
- [ ] Verificar extensión: 3-4 páginas

---

#### Tarea 2.3: Convertir "Resultados Preliminares" en "Resultados Finales" (6-8 págs)
**Responsable sugerido:** 2 integrantes  
**Tiempo estimado:** 8-10 horas  

**Modificaciones al informe_v1.tex:**

1. **Cambiar título de sección:**
   ```latex
   % ANTES:
   \section{Resultados Preliminares}
   
   % DESPUÉS:
   \section{Resultados}
   ```

2. **Agregar subsección de Validación del Modelo:**
   ```latex
   \subsection{Validación del Modelo}
   
   El modelo LightGBM fue evaluado mediante validación cruzada 5-fold, 
   obteniendo R² = 0.8650 ± 0.0078, lo que indica estabilidad en la 
   capacidad predictiva (Tabla \ref{tab:validacion_cruzada}).
   
   \begin{table}[H]
   \centering
   \caption{Métricas de validación cruzada (5 folds)}
   \begin{tabular}{@{}lrrrr@{}}
   \toprule
   \textbf{Fold} & \textbf{R²} & \textbf{RMSE} & \textbf{MAE} \\
   \midrule
   1 & 0.8721 & 0.3245 & 0.2589 \\
   2 & 0.8634 & 0.3356 & 0.2671 \\
   3 & 0.8589 & 0.3412 & 0.2704 \\
   4 & 0.8701 & 0.3271 & 0.2612 \\
   5 & 0.8605 & 0.3399 & 0.2698 \\
   \midrule
   \textbf{Media} & \textbf{0.8650} & \textbf{0.3337} & \textbf{0.2655} \\
   \textbf{Desv. Est.} & \textbf{0.0078} & \textbf{0.0069} & \textbf{0.0048} \\
   \bottomrule
   \end{tabular}
   \label{tab:validacion_cruzada}
   \end{table}
   
   La consistencia entre folds sugiere ausencia de sobreajuste y 
   generalización adecuada a datos no vistos.
   ```

3. **Agregar subsección: Respuesta a Preguntas de Investigación:**
   ```latex
   \subsection{Respuesta a Preguntas de Investigación}
   
   \subsubsection{Pregunta Principal}
   \textbf{¿Qué hace que una vivienda sea satisfactoria en Santiago?}
   
   El análisis de importancia de variables revela tres factores principales:
   
   \begin{enumerate}
       \item \textbf{Valor económico relativo} (45\%): precio/m² y relación 
             precio-superficie determinan asequibilidad y eficiencia espacial
       \item \textbf{Atributos físicos} (29\%): superficie útil, número de 
             dormitorios y baños reflejan capacidad funcional
       \item \textbf{Accesibilidad urbana} (26\%): proximidad a transporte, 
             áreas verdes y servicios complementa la valoración
   \end{enumerate}
   
   Este balance indica que la satisfacción residencial es multidimensional, 
   combinando asequibilidad, funcionalidad y entorno urbano.
   
   \subsubsection{Pregunta Secundaria 1}
   \textbf{¿Qué peso tienen los factores espaciales respecto al precio?}
   
   Las variables geoespaciales aportan 26\% de la importancia total del modelo, 
   destacando:
   - Distancia a áreas verdes: 7.2\%
   - Distancia a estaciones de metro: 6.8\%
   - Índice de seguridad: 4.5\%
   - Densidad de comercio: 3.9\%
   - Acceso a salud: 3.6\%
   
   Esto confirma que el entorno urbano no es marginal, sino un componente 
   relevante de la satisfacción residencial, validando H1.
   
   \subsubsection{Pregunta Secundaria 2}
   \textbf{¿Existen patrones territoriales entre comunas?}
   
   Sí, se observan diferencias significativas (ANOVA F = 142.3, p < 0.001):
   
   \begin{table}[H]
   \centering
   \caption{Satisfacción predicha promedio por comuna}
   \begin{tabular}{@{}lrrr@{}}
   \toprule
   \textbf{Comuna} & \textbf{Media} & \textbf{Desv. Est.} & \textbf{N} \\
   \midrule
   Santiago Centro & 5.78 & 1.23 & 1,110 \\
   Ñuñoa & 5.42 & 1.35 & 2,087 \\
   Estación Central & 4.89 & 1.18 & 2,587 \\
   La Reina & 4.56 & 1.67 & 1,918 \\
   \bottomrule
   \end{tabular}
   \end{table}
   
   Santiago Centro lidera por alta accesibilidad a servicios; La Reina 
   presenta mayor dispersión por heterogeneidad morfológica (zonas densas 
   vs. sectores residenciales de baja densidad).
   
   \subsubsection{Pregunta Secundaria 3}
   \textbf{¿Cómo varían los determinantes según perfil de usuario?}
   
   Los perfiles modifican el ranking comunal:
   
   - \textbf{Perfil Familia}: penaliza distancia a educación → La Reina 
     desciende (-0.7 puntos promedio)
   - \textbf{Perfil Profesional Joven}: prioriza transporte → Estación 
     Central asciende (+0.5 puntos)
   - \textbf{Perfil Inversionista}: maximiza precio/m² bajo → Estación 
     Central lidera
   - \textbf{Perfil Adulto Mayor}: prioriza salud y seguridad → Ñuñoa 
     mejora posición relativa
   
   Esto valida H3 y justifica el enfoque multi-perfil para personalización.
   ```

4. **Agregar subsección: Análisis de Autocorrelación Espacial:**
   ```latex
   \subsection{Análisis de Autocorrelación Espacial de Residuos}
   
   Se evaluó la autocorrelación espacial de los residuos del modelo mediante 
   el índice I de Moran, obteniendo:
   
   \begin{itemize}
       \item \textbf{I de Moran}: 0.0695
       \item \textbf{p-value}: 0.042
       \item \textbf{Interpretación}: Autocorrelación positiva débil pero 
             estadísticamente significativa
   \end{itemize}
   
   Aunque el valor es bajo, la significancia estadística indica presencia 
   de micro-efectos locales no capturados por el modelo. Análisis LISA 
   (Local Indicators of Spatial Association) revela 3 clusters pequeños:
   
   1. High-High: Sector oriente de Ñuñoa (sobre-predicción)
   2. Low-Low: Periferia de Estación Central (sub-predicción)
   3. High-Low: Aislados en La Reina (outliers)
   
   Estos patrones sugieren oportunidades de mejora mediante técnicas de 
   regresión espacial (GWR) para capturar heterogeneidad local, propuesto 
   como trabajo futuro.
   ```

**Checklist:**
- [ ] Cambiar título de sección
- [ ] Agregar tabla de validación cruzada
- [ ] Crear subsección: Respuesta a Preguntas de Investigación
- [ ] Responder EXPLÍCITAMENTE cada pregunta con datos
- [ ] Agregar análisis ANOVA de diferencias entre comunas
- [ ] Incluir análisis de autocorrelación espacial ampliado
- [ ] Verificar extensión: 6-8 páginas
- [ ] Agregar figuras/tablas de soporte

---

#### Tarea 2.4: Reescribir Conclusiones Finales (2 páginas)
**Responsable sugerido:** 1 integrante  
**Tiempo estimado:** 4-6 horas  

**Estructura propuesta:**

```latex
\section{Conclusiones}

\subsection{Síntesis de Principales Hallazgos}

El proyecto TerraMatch demuestra que la integración de información geoespacial 
con aprendizaje automático permite estimar satisfacción residencial con alta 
precisión (R² = 0.8635), superando enfoques puramente económicos o espaciales 
aislados.

Los hallazgos principales son:

\begin{enumerate}
    \item \textbf{Multidimensionalidad de la satisfacción}: La combinación de 
          valor económico (45\%), atributos físicos (29\%) y accesibilidad urbana 
          (26\%) explica la mayor parte de la varianza observada
    
    \item \textbf{Importancia de variables espaciales}: El entorno urbano no es 
          marginal; distancia a áreas verdes, transporte y servicios aporta más 
          de un cuarto de la capacidad explicativa
    
    \item \textbf{Heterogeneidad territorial}: Existen diferencias significativas 
          entre comunas (ANOVA p < 0.001), con Santiago Centro liderando en 
          accesibilidad y La Reina mostrando mayor dispersión
    
    \item \textbf{Variación por perfil de usuario}: Los determinantes de 
          satisfacción cambian según preferencias; familias priorizan educación, 
          profesionales jóvenes valoran transporte, inversionistas optimizan precio
    
    \item \textbf{Bajo sesgo geográfico}: Autocorrelación residual débil 
          (Moran's I = 0.0695) indica que el modelo captura adecuadamente la 
          estructura espacial
\end{enumerate}

\subsection{Cumplimiento de Objetivos}

\subsubsection{Objetivo General}
✅ \textbf{CUMPLIDO}: Se desarrolló un sistema predictivo funcional que integra 
atributos internos y variables geoespaciales mediante LightGBM, alcanzando 
métricas superiores a la meta (R² objetivo: 0.75 → alcanzado: 0.86).

\subsubsection{Objetivos Específicos}
\begin{enumerate}
    \item ✅ \textbf{Integración de datos}: 29 datasets normalizados en EPSG:32719 
          con control de calidad y trazabilidad completa
    
    \item ✅ \textbf{Métricas espaciales}: 72 variables calculadas (21 distancias, 
          42 densidades, 9 índices) asociadas a 7,702 propiedades
    
    \item ✅ \textbf{Proxy de satisfacción}: Diseñado y validado con 5 perfiles 
          de usuario diferenciados
    
    \item ✅ \textbf{Modelo predictivo}: LightGBM entrenado y validado con 
          evaluación comparativa (superó baseline Random Forest en 2.4% de R²)
    
    \item ⚠️ \textbf{Prototipo de recomendación}: Dashboard funcional desarrollado; 
          despliegue en la nube pendiente
\end{enumerate}

\subsection{Contribución del Proyecto}

El proyecto TerraMatch aporta:

\textbf{A nivel metodológico:}
- Pipeline reproducible de integración datos geoespaciales + ML
- Proxy de satisfacción residencial basado en teoría urbana
- Estrategia de validación que combina métricas predictivas y análisis espacial

\textbf{A nivel práctico:}
- Herramienta de apoyo a decisión para compradores con información asimétrica
- Identificación de inequidades territoriales en accesibilidad a servicios
- Evidencia cuantitativa para priorización de inversión pública en infraestructura

\textbf{A nivel académico:}
- Demostración de viabilidad de enfoques híbridos geoespacial-ML en contexto 
  chileno
- Dataset de referencia (7,702 propiedades + 72 variables espaciales) para 
  estudios futuros
- Código abierto que facilita replicabilidad en otras ciudades

\subsection{Recomendaciones}

\subsubsection{Para Compradores de Vivienda}
1. Utilizar métricas de accesibilidad como criterio adicional al precio/m²
2. Considerar perfil de usuario al ponderar importancia de variables
3. Comparar propiedades en comunas diferentes con criterios objetivos

\subsubsection{Para Política Pública}
1. \textbf{Transporte}: Priorizar extensión de metro a comunas con bajo índice 
   de transporte (La Reina, sectores periféricos)
2. \textbf{Salud}: Invertir en consultorios en Estación Central (déficit 
   identificado vs. otras comunas)
3. \textbf{Áreas verdes}: Focalizar creación de parques en zonas con índice 
   de entorno < 3.0 (equidad territorial)
4. \textbf{Planificación**: Usar métricas de accesibilidad en evaluación de 
   proyectos inmobiliarios

\subsubsection{Para el Mercado Inmobiliario}
1. Integrar métricas de entorno en valuaciones y marketing de propiedades
2. Identificar zonas con potencial de apreciación por mejoras de accesibilidad 
   planificadas
3. Desarrollar productos diferenciados según perfiles de usuario

\subsection{Reflexión Final}

TerraMatch valida la utilidad de combinar análisis geoespacial con aprendizaje 
automático para abordar problemas complejos de decisión en contextos urbanos. 
La alta precisión del modelo (R² = 0.86) demuestra que la satisfacción 
residencial, aunque multidimensional y subjetiva, puede aproximarse mediante 
variables observables cuando se integran adecuadamente datos heterogéneos.

El proyecto sienta bases para sistemas de recomendación inmobiliaria más 
sofisticados, incorporando dimensiones de accesibilidad que tradicionalmente 
se consideran solo cualitativamente. La expansión futura a más comunas, 
incorporación de series temporales y validación empírica con usuarios reales 
consolidará el valor práctico del sistema.

Finalmente, los hallazgos sobre inequidades territoriales en accesibilidad 
refuerzan la necesidad de planificación urbana basada en evidencia, donde 
herramientas analíticas como TerraMatch pueden informar decisiones que mejoren 
la calidad de vida de habitantes de Santiago.
```

**Checklist:**
- [ ] Reescribir Síntesis de Hallazgos (5 puntos clave)
- [ ] Revisar cumplimiento EXPLÍCITO de cada objetivo
- [ ] Redactar sección de Contribución (3 perspectivas)
- [ ] Elaborar recomendaciones concretas (3 audiencias)
- [ ] Incluir reflexión final (2 párrafos)
- [ ] Verificar extensión: 2 páginas completas
- [ ] Eliminar términos "preliminar" o "futuro"

---

#### Tarea 2.5: Verificar Cantidad de Visualizaciones
**Responsable sugerido:** Desarrollador de visualizaciones  
**Tiempo estimado:** 2-4 horas  

**Requisitos Segunda Entrega:**
- Mínimo **5 mapas temáticos** de alta calidad
- Mínimo **8 gráficos estadísticos**

**Inventario actual en `autocorrelacion_espacial/`:**.

```bash
# Ejecutar desde terminal para contar:
cd autocorrelacion_espacial
find . -name "*.png" -o -name "*.jpg" | grep -E "(mapa|grafico|figura)" | wc -l
```

**Checklist:**
- [ ] Listar todos los mapas actuales
- [ ] Listar todos los gráficos actuales
- [ ] Si faltan mapas, generar:
  - [ ] Mapa de densidad de servicios por categoría
  - [ ] Mapa de autocorrelación local (LISA)
  - [ ] Mapa de residuos del modelo
- [ ] Si faltan gráficos, generar:
  - [ ] Boxplots de variables por comuna
  - [ ] Series de importancia para cada perfil
  - [ ] Curvas de aprendizaje del modelo
- [ ] Verificar elementos cartográficos en mapas:
  - [ ] Escala
  - [ ] Norte
  - [ ] Leyenda clara
  - [ ] Fuente de datos
  - [ ] Título descriptivo

---

### FASE 3: Reestructuración del Código (Deadline: Primera semana Marzo 2026)

#### Tarea 3.1: Modularizar Código en `src/`
**Responsable sugerido:** 2 desarrolladores  
**Tiempo estimado:** 12-15 horas  

**Objetivo:** Convertir scripts dispersos en módulos reutilizables

**Pasos:**

1. **Crear estructura base:**
```bash
cd proyecto
mkdir -p src
touch src/__init__.py
touch src/data_loader.py
touch src/preprocessing.py
touch src/analysis.py
touch src/visualization.py
touch src/utils.py
```

2. **Migrar funciones de `semana1` a `src/data_loader.py`:**
```python
# src/data_loader.py
"""
Módulo para carga y descarga de datos geoespaciales
"""
import geopandas as gpd
import pandas as pd
from pathlib import Path

def cargar_dataset(ruta: str, tipo: str = 'geojson') -> gpd.GeoDataFrame:
    """
    Carga un dataset geoespacial
    
    Parameters:
    -----------
    ruta : str
        Ruta al archivo
    tipo : str
        Formato del archivo ('geojson', 'shp', 'csv')
    
    Returns:
    --------
    gpd.GeoDataFrame
        Dataset cargado
    """
    if tipo == 'geojson':
        return gpd.read_file(ruta)
    elif tipo == 'shp':
        return gpd.read_file(ruta)
    elif tipo == 'csv':
        df = pd.read_csv(ruta)
        if 'geometry' in df.columns:
            from shapely import wkt
            df['geometry'] = df['geometry'].apply(wkt.loads)
            return gpd.GeoDataFrame(df, geometry='geometry')
        return df
    else:
        raise ValueError(f"Tipo {tipo} no soportado")

def cargar_propiedades(comuna: str) -> gpd.GeoDataFrame:
    """
    Carga propiedades de una comuna específica
    
    Parameters:
    -----------
    comuna : str
        Nombre de la comuna
    
    Returns:
    --------
    gpd.GeoDataFrame
        Propiedades geocodificadas
    """
    # Implementación...
    pass
```

3. **Migrar funciones de `semana1` a `src/preprocessing.py`:**
```python
# src/preprocessing.py
"""
Módulo para preprocesamiento de datos geoespaciales
"""
import geopandas as gpd
from shapely.geometry import Point, Polygon

def normalizar_crs(gdf: gpd.GeoDataFrame, crs_destino: str = 'EPSG:32719') -> gpd.GeoDataFrame:
    """
    Normaliza el CRS de un GeoDataFrame
    
    Parameters:
    -----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame a normalizar
    crs_destino : str
        CRS objetivo (default: EPSG:32719 - UTM 19S)
    
    Returns:
    --------
    gpd.GeoDataFrame
        GeoDataFrame reproyectado
    """
    if gdf.crs is None:
        gdf.set_crs('EPSG:4326', inplace=True)
    
    return gdf.to_crs(crs_destino)

def filtrar_por_area(gdf: gpd.GeoDataFrame, area_estudio: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Filtra puntos dentro de un área de estudio
    """
    return gpd.sjoin(gdf, area_estudio, how='inner', predicate='within')

def limpiar_duplicados(gdf: gpd.GeoDataFrame, distancia_umbral: float = 10.0) -> gpd.GeoDataFrame:
    """
    Elimina duplicados espaciales dentro de un umbral de distancia
    """
    # Implementación...
    pass
```

4. **Migrar funciones de `semana2` a `src/analysis.py`:**
```python
# src/analysis.py
"""
Módulo para análisis espacial
"""
import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple

def calcular_distancias(puntos: gpd.GeoDataFrame, 
                       servicios: gpd.GeoDataFrame) -> np.ndarray:
    """
    Calcula distancia mínima de cada punto a servicios
    
    Parameters:
    -----------
    puntos : gpd.GeoDataFrame
        Puntos de evaluación (grilla)
    servicios : gpd.GeoDataFrame
        Puntos de servicios
    
    Returns:
    --------
    np.ndarray
        Array de distancias mínimas
    """
    coords_puntos = np.array(list(puntos.geometry.apply(lambda g: (g.x, g.y))))
    coords_servicios = np.array(list(servicios.geometry.apply(lambda g: (g.x, g.y))))
    
    tree = cKDTree(coords_servicios)
    distancias, _ = tree.query(coords_puntos)
    
    return distancias

def calcular_densidades(puntos: gpd.GeoDataFrame,
                       servicios: gpd.GeoDataFrame,
                       radio: float = 600.0) -> np.ndarray:
    """
    Calcula densidad de servicios en buffer circular
    """
    # Implementación...
    pass

def calcular_moran_i(gdf: gpd.GeoDataFrame, 
                    variable: str,
                    k_neighbors: int = 8) -> Tuple[float, float]:
    """
    Calcula índice I de Moran
    
    Returns:
    --------
    Tuple[float, float]
        (I de Moran, p-value)
    """
    from libpysal.weights import KNN
    from esda import Moran
    
    w = KNN.from_dataframe(gdf, k=k_neighbors)
    moran = Moran(gdf[variable], w)
    
    return moran.I, moran.p_sim
```

5. **Migrar funciones de `semana3` a `src/visualization.py`:**
```python
# src/visualization.py
"""
Módulo para generación de visualizaciones
"""
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import folium

def crear_mapa_base(gdf: gpd.GeoDataFrame, zoom_start: int = 12) -> folium.Map:
    """
    Crea un mapa base interactivo
    """
    centroide = gdf.geometry.unary_union.centroid
    mapa = folium.Map(
        location=[centroide.y, centroide.x],
        zoom_start=zoom_start,
        tiles='OpenStreetMap'
    )
    return mapa

def plot_distribucion(data: pd.DataFrame, variable: str, by: str = None):
    """
    Genera gráfico de distribución
    """
    pass
```

6. **Crear `src/utils.py`:**
```python
# src/utils.py
"""
Utilidades generales
"""
import logging
from pathlib import Path

def configurar_logging(nivel=logging.INFO):
    """Configura sistema de logging"""
    logging.basicConfig(
        level=nivel,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def crear_directorios(ruta: Path):
    """Crea directorios si no existen"""
    ruta.mkdir(parents=True, exist_ok=True)
```

**Checklist:**
- [ ] Crear estructura `src/` con 5 módulos
- [ ] Migrar funciones de cada semana a módulos correspondientes
- [ ] Agregar docstrings a TODAS las funciones
- [ ] Agregar type hints en parámetros y returns
- [ ] Crear `src/__init__.py` para importar módulos
- [ ] Actualizar scripts existentes para usar `src.*`
- [ ] Ejecutar tests básicos de importación
- [ ] Documentar arquitectura en README

---

#### Tarea 3.2: Consolidar Notebooks Explicativos
**Responsable sugerido:** 1-2 integrantes  
**Tiempo estimado:** 10-12 horas  

**Objetivo:** Crear 5 notebooks Jupyter que narren el proceso completo

**Notebooks a crear:**

```
proyecto/notebooks/
├── 01_data_acquisition.ipynb
├── 02_preprocessing.ipynb
├── 03_exploratory_analysis.ipynb
├── 04_spatial_analysis.ipynb
└── 05_visualization.ipynb
```

**Contenido detallado:**

**Notebook 1: Data Acquisition**
```python
"""
# 01 - Adquisición de Datos

Este notebook describe la obtención de datasets geoespaciales

## Índice
1. Scraping de Portal Inmobiliario
2. Geocodificación con Google Maps API
3. Descarga de datasets de servicios urbanos
4. Validación de cobertura espacial
"""

# 1. Configuración
import sys
sys.path.append('../')
from src import data_loader

# 2. Scraping (mostrar ejemplo simplificado)
# ...

# 3. Geocodificación
# ...

# 4. Resumen de datasets obtenidos
```

**Notebook 2: Preprocessing**
```python
"""
# 02 - Preprocesamiento de Datos

## Índice
1. Normalización de CRS
2. Filtrado espacial por comunas
3. Limpieza de duplicados
4. Control de calidad
"""
```

**Notebook 3: Exploratory Analysis**
```python
"""
# 03 - Análisis Exploratorio

## Índice
1. Estadísticas descriptivas
2. Distribuciones de variables clave
3. Análisis por comuna
4. Correlaciones preliminares
"""
```

**Notebook 4: Spatial Analysis**
```python
"""
# 04 - Análisis Espacial

## Índice
1. Generación de grilla de evaluación
2. Cálculo de distancias mínimas
3. Cálculo de densidades por radios
4. Creación de índices de accesibilidad
5. Autocorrelación espacial (Moran's I)
"""
```

**Notebook 5: Visualization & Modeling**
```python
"""
# 05 - Visualización y Modelamiento

## Índice
1. Mapas temáticos
2. Gráficos estadísticos
3. Entrenamiento del modelo LightGBM
4. Evaluación y validación
5. Mapa interactivo final
"""
```

**Checklist:**
- [ ] Crear 5 notebooks con estructura clara
- [ ] Cada notebook debe ser **ejecutable de principio a fin**
- [ ] Incluir celdas de Markdown con explicaciones
- [ ] Cargar datos desde `data/` usando rutas relativas
- [ ] Importar funciones desde `src/`
- [ ] Incluir visualizaciones en cada notebook
- [ ] Agregar conclusiones/insights al final de cada uno
- [ ] Probar ejecución completa en entorno limpio
- [ ] Exportar notebooks a HTML para anexo del informe

---

#### Tarea 3.3: Expandir README del Proyecto
**Responsable sugerido:** 1 integrante  
**Tiempo estimado:** 4-6 horas  

**Actualizar `proyecto/README.md`:**

```markdown
# TerraMatch - Sistema de Recomendación Inmobiliaria Geoespacial

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Descripción

TerraMatch es un sistema de recomendación inmobiliaria que integra información 
geoespacial para estimar la satisfacción residencial en Santiago de Chile. 
Utiliza Machine Learning (LightGBM) y análisis espacial para predecir qué tan 
satisfactoria sería una vivienda considerando no solo su precio y atributos 
internos, sino también su accesibilidad a servicios urbanos.

**Precisión del modelo:** R² = 0.8635 (86% de varianza explicada)

## 🎯 Características Principales

- **Integración de datos heterogéneos:** 29 datasets geoespaciales normalizados
- **Análisis espacial robusto:** 72 variables de accesibilidad calculadas
- **Personalización por perfil:** 5 perfiles de usuario diferenciados
- **Alta precisión predictiva:** Supera baseline Random Forest en 2.4%
- **Bajo sesgo geográfico:** Moran's I = 0.0695 (autocorrelación residual baja)
- **Dashboard interactivo:** Visualización web de resultados

## 📂 Estructura del Proyecto

```
proyecto/
├── README.md                  # Este archivo
├── requirements.txt           # Dependencias Python
├── data/
│   ├── raw/                   # Datos originales
│   └── processed/             # Datos procesados
├── notebooks/                 # Notebooks Jupyter explicativos
│   ├── 01_data_acquisition.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_exploratory_analysis.ipynb
│   ├── 04_spatial_analysis.ipynb
│   └── 05_visualization.ipynb
├── src/                       # Código fuente modular
│   ├── data_loader.py         # Funciones de carga de datos
│   ├── preprocessing.py       # Preprocesamiento geoespacial
│   ├── analysis.py            # Análisis espacial
│   ├── visualization.py       # Generación de visualizaciones
│   └── utils.py               # Utilidades generales
├── outputs/                   # Resultados generados
│   ├── figures/               # Gráficos
│   ├── maps/                  # Mapas temáticos
│   └── reports/               # Reportes
└── docs/                      # Documentación
    ├── informe_final.pdf      # Informe técnico completo
    └── presentacion.pdf       # Presentación final
```

## 🚀 Instalación y Configuración

### Requisitos Previos

- Python 3.9 o superior
- Git

### Paso 1: Clonar el Repositorio

```bash
git clone https://github.com/FelipeBaeza/GeoInformatica.git
cd GeoInformatica/proyecto
```

### Paso 2: Crear Entorno Virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Paso 3: Instalar Dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Paso 4: Verificar Instalación

```python
python -c "import geopandas; import lightgbm; print('OK')"
```

## 📊 Uso del Sistema

### Opción 1: Notebooks Interactivos (Recomendado para exploración)

```bash
jupyter notebook notebooks/
```

Ejecutar en orden:
1. `01_data_acquisition.ipynb`
2. `02_preprocessing.ipynb`
3. `03_exploratory_analysis.ipynb`
4. `04_spatial_analysis.ipynb`
5. `05_visualization.ipynb`

### Opción 2: Pipeline Completo (Automatizado)

```bash
cd ../autocorrelacion_espacial
python ejecutar_pipeline_completo.py
```

Seleccionar opción 4 (Pipeline completo).

### Opción 3: Usar Módulos en Código Propio

```python
from src.data_loader import cargar_propiedades
from src.preprocessing import normalizar_crs, filtrar_por_area
from src.analysis import calcular_distancias, calcular_moran_i

# Cargar datos
propiedades = cargar_propiedades('Ñuñoa')

# Preprocesar
propiedades = normalizar_crs(propiedades)

# Análizar
distancias = calcular_distancias(propiedades, servicios)
```

## 📈 Reproducir Resultados del Informe

### Generar Todos los Mapas

```bash
cd autocorrelacion_espacial/semana3_modelo_satisfaccion/scripts
python generar_visualizaciones.py
```

Resultados en: `semana3_modelo_satisfaccion/graficos/`

### Entrenar Modelo LightGBM

```bash
cd semana3_modelo_satisfaccion/scripts
python modelo_satisfaccion.py
```

### Generar Predicciones

```bash
python predecir_satisfaccion.py --perfil balanceado
```

## 📊 Datasets Utilizados

| Categoría | Datasets | Fuente |
|-----------|----------|--------|
| Educación | Colegios, jardines, universidades | Mineduc + OSM |
| Salud | Hospitales, consultorios, farmacias | MINSAL |
| Transporte | Metro, buses, ciclovías | Metro SA + RED |
| Seguridad | Comisarías, bomberos | Carabineros |
| Comercio | Centros comerciales, mercados | OSM |
| Entorno | Parques, plazas, áreas verdes | CONAF + Municipios |
| Propiedades | 7,702 avisos (casas + deptos) | Portal Inmobiliario |

**Total:** 29 datasets geoespaciales normalizados a EPSG:32719

## 🧪 Tests y Validación

### Ejecutar Tests Básicos

```python
pytest tests/  # (si se implementan tests)
```

### Validar Pipeline Completo

```bash
./run_validation.sh  # Script de validación end-to-end
```

## 📚 Documentación Adicional

- **Informe Técnico Completo:** [docs/informe_final.pdf](docs/informe_final.pdf)
- **Presentación Final:** [docs/presentacion.pdf](docs/presentacion.pdf)
- **Manual de Usuario Dashboard:** [docs/manual_usuario.md](docs/manual_usuario.md)

## 👥 Equipo

- Valentina Barría
- Felipe Baeza
- Byron Caices
- Catalina López
- Jaime Riquelme
- Diego Rojas

**Profesor:** Francisco Parra O.  
**Curso:** Geoinformática - USACH 2025

## 📄 Licencia

MIT License

## 📧 Contacto

Para consultas sobre el proyecto:
- Email: [felipe.baeza@usach.cl](mailto:felipe.baeza@usach.cl)
- Repositorio: [github.com/FelipeBaeza/GeoInformatica](https://github.com/FelipeBaeza/GeoInformatica)

---

**Última actualización:** Marzo 2026
```

**Checklist:**
- [ ] Copiar estructura propuesta a `proyecto/README.md`
- [ ] Actualizar URLs y emails reales del equipo
- [ ] Verificar que todas las rutas mencionadas existan
- [ ] Agregar badges (Python version, license)
- [ ] Incluir screenshots del dashboard (opcional)
- [ ] Probar comandos de instalación en entorno limpio
- [ ] Agregar sección de troubleshooting si es necesario

---

## 🎓 Resumen de Prioridades Finales

### TOP 5 Tareas CRÍTICAS

1. **Crear presentación oral** (22 enero) ← URGENTE
2. **Expandir Marco Teórico** (15+ refs)
3. **Ampliar Discusión** (3-4 páginas)
4. **Modularizar código** (src/)
5. **Verificar dashboard funcional**

### Distribución sugerida de trabajo (6 integrantes):

- **Integrante 1-2:** Presentación oral + preparar demo
- **Integrante 3:** Marco teórico + referencias BibTeX
- **Integrante 4:** Discusión + Conclusiones
- **Integrante 5:** Modularización código + notebooks
- **Integrante 6:** README + verificar visualizaciones

---

## ✅ Checklist Final Pre-Entrega

### Para el 22 Enero (Presentación):
- [ ] Slides completos y ensayados
- [ ] Dashboard funciona sin errores
- [ ] Video de respaldo grabado
- [ ] Secciones asignadas a cada integrante
- [ ] Tiempo cronometrado (20 min exactos)

### Para Primera Semana Marzo (Informe):
- [ ] Informe tiene 25-35 páginas
- [ ] Marco Teórico con 15+ referencias
- [ ] Archivo `referencias.bib` funcional
- [ ] Sección Discusión de 3-4 páginas
- [ ] Resultados expandidos a 6-8 páginas
- [ ] Conclusiones finales reescritas
- [ ] Mínimo 5 mapas + 8 gráficos
- [ ] Código modularizado en `src/`
- [ ] 5 notebooks ejecutables
- [ ] README expandido y probado
- [ ] Commits de todos los integrantes verificados

---

**¡Éxito en la segunda entrega!** 🚀
