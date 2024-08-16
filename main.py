from fastapi import FastAPI, Query
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Datos
movies = pd.read_parquet('./Datasets/movies.parquet')
casting = pd.read_parquet('./Datasets/union_mcasting.parquet')
crew = pd.read_parquet('./Datasets/union_mcrew.parquet')

# Asegurarse de que release_date esté en formato de fecha
movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce')
# Asegurarse de que popularity esté en formato numérico
movies['popularity'] = pd.to_numeric(movies['popularity'], errors='coerce')

app = FastAPI(
    title="Proyecto Recomendacion Peliculas",
    version="1.0.0 / Andres Acosta",
    openapi_tags=[{
        "name": "Data Sceincie",
    }]
)



@app.get("/movies per month", tags=["películas por mes"])
def cantidad_filmaciones_mes(mes):
    # Crear un diccionario para mapear nombres de meses en español con números
    meses = {
        'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4,
        'mayo': 5, 'junio': 6, 'julio': 7, 'agosto': 8,
        'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
    }
    
    # Convertir el mes ingresado a su correspondiente número
    mes_numero = meses.get(mes.lower())
    
    if not mes_numero:
        return "Mes ingresado no es válido."
    
    # Filtrar el dataframe por el mes
    cantidad = movies[movies['release_date'].dt.month == mes_numero].shape[0]
    
    return f"{cantidad} cantidad de películas fueron estrenadas en el mes de {mes.capitalize()}."



@app.get("/title, year of release, score", tags=["películas por dia"])
def cantidad_filmaciones_dia(dia):
    # Crear un diccionario para mapear nombres de días en español con números
    dias_semana = {
        'lunes': 0, 'martes': 1, 'miércoles': 2, 'miercoles': 2, 'jueves': 3,
        'viernes': 4, 'sábado': 5, 'sabado': 5, 'domingo': 6
    }
    
    # Convertir el día ingresado a su correspondiente número
    dia_numero = dias_semana.get(dia.lower())
    
    if dia_numero is None:
        return "Día ingresado no es válido."
    
    # Filtrar el dataframe por el día de la semana
    cantidad = movies[movies['release_date'].dt.dayofweek == dia_numero].shape[0]
    
    return f"{cantidad} cantidad de películas fueron estrenadas en los días {dia.capitalize()}."



@app.get("/movies per day", tags=["Título, año de lanzamiento, puntaje"])
def score_titulo(titulo_de_la_filmacion):
    # Filtrar el DataFrame por el título de la filmación
    pelicula = movies[movies['title'].str.lower() == titulo_de_la_filmacion.lower()]
    
    if pelicula.empty:
        return f"No se encontró ninguna película con el título '{titulo_de_la_filmacion}'."
    
    # Extraer la información relevante
    titulo = pelicula.iloc[0]['title']
    anio_estreno = pelicula.iloc[0]['release_date'].year
    score = round(pelicula.iloc[0]['popularity'], 2) # Suponiendo que 'popularity' es la columna de score
    
    return f"La película {titulo} fue estrenada en el año {anio_estreno} con un score/popularidad de {score}"