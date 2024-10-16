from fastapi import FastAPI, Query
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack


# Datos
movies = pd.read_parquet('Datasets/movies_compressed.parquet')

# Asegurarse de que release_date esté en formato de fecha
movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce')

# Asegurarse de que popularity esté en formato numérico
movies['popularity'] = pd.to_numeric(movies['popularity'], errors='coerce')

app = FastAPI(
    title="Proyecto Recomendacion Peliculas",
    version="1.0.0 / Andres Acosta",
    openapi_tags=[{
        "name": "Data Science",
    }]
)

@app.get("/movies per month", tags=["Películas por mes"])

def cantidad_filmaciones_mes(mes: str = Query(default= 'enero')):
    """
    <strong>Esta función devuelve la cantidad de películas que se estrenaron en un mes específico.<strong>

    Argumentos:

            El nombre del mes en español.

        Retorna:

            La cantidad de películas que se estrenaron en el mes especificado.
    """
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
    
    # Eliminar duplicados basado en el título
    movies_sin_duplicados = movies.drop_duplicates(subset='title')
    
    # Filtrar el dataframe por el mes
    cantidad = movies_sin_duplicados[movies_sin_duplicados['release_date'].dt.month == mes_numero].shape[0]
    
    return f"{cantidad} cantidad de películas fueron estrenadas en el mes de {mes.capitalize()}."

#----------------------------------------------------------------------------------------------------------

@app.get("/title, year of release, score", tags=["Películas por dia"])

def cantidad_filmaciones_dia(dia: str = Query(default= 'lunes')):
    """
    <strong>Esta función devuelve la cantidad de películas que se estrenaron en un día específico.<strong>

    Argumentos:

            El nombre del día de la semana en español.

        Retorna:

            La cantidad de películas que se estrenaron en el día especificado.
    """
    # Crear un diccionario para mapear nombres de días en español con números
    dias_semana = {
        'lunes': 0, 'martes': 1, 'miércoles': 2, 'miercoles': 2, 'jueves': 3,
        'viernes': 4, 'sábado': 5, 'sabado': 5, 'domingo': 6
    }
    
    # Convertir el día ingresado a su correspondiente número
    dia_numero = dias_semana.get(dia.lower())
    
    if dia_numero is None:
        return "Día ingresado no es válido."
    
    movies_sin_duplicados = movies.drop_duplicates(subset='title')
    
    # Filtrar el dataframe por el día de la semana
    cantidad = movies_sin_duplicados[movies_sin_duplicados['release_date'].dt.dayofweek == dia_numero].shape[0]
    
    return f"{cantidad} cantidad de películas fueron estrenadas en los días {dia.capitalize()}."

#----------------------------------------------------------------------------------------------------------

@app.get("/movies per day", tags=["Título, Año de lanzamiento, Puntaje"])

def score_titulo(titulo_de_la_filmacion: str = Query(default= 'Toy Story')):

    """
    <strong>Esta función devuelve el título, año de lanzamiento y score de una película específica.<strong>

    Argumentos:

            El título de la película.

        Retorna:

            El título, año de lanzamiento y score de la película especificada.
    """
    # Filtrar el DataFrame por el título de la filmación
    pelicula = movies[movies['title'].str.lower() == titulo_de_la_filmacion.lower()]
    
    if pelicula.empty:
        return f"No se encontró ninguna película con el título '{titulo_de_la_filmacion}'."
    
    # Extraer la información relevante
    titulo = pelicula.iloc[0]['title']
    anio_estreno = pelicula.iloc[0]['release_date'].year
    score = round(pelicula.iloc[0]['popularity'], 2) # Suponiendo que 'popularity' es la columna de score
    
    return f"La película {titulo} fue estrenada en el año {anio_estreno} con un score/popularidad de {score}"

#----------------------------------------------------------------------------------------------------------

@app.get("/ratings, average", tags=["Cantidad de votos y promedio"])

def votos_titulo(titulo_de_la_filmacion: str = Query(default= 'Toy Story')):

    """
    <strong>Esta función devuelve la cantidad de votos y el promedio de las votaciones de una película específica.<strong>

    Argumentos:

            El título de la película.

        Retorna:

            La cantidad de votos y el promedio de las votaciones de la película especificada.
    """
    # Filtrar el DataFrame por el título de la filmación
    pelicula = movies[movies['title'].str.lower() == titulo_de_la_filmacion.lower()]
    
    if pelicula.empty:
        return f"No se encontró ninguna película con el título '{titulo_de_la_filmacion}'."
    
    # Extraer la cantidad de votos y el promedio de las votaciones
    votos = pelicula.iloc[0]['vote_count']
    promedio_votos = pelicula.iloc[0]['vote_average']
    
    if votos < 2000:
        return f"La película {pelicula.iloc[0]['title']} no cumple con el mínimo de 2000 valoraciones. Tiene {votos} valoraciones."

    # Extraer el título y el año de estreno
    titulo = pelicula.iloc[0]['title']
    anio_estreno = pelicula.iloc[0]['release_date'].year
    
    return f"La película {titulo} fue estrenada en el año {anio_estreno}. La misma cuenta con un total de {votos} valoraciones, con un promedio de {promedio_votos}"

#----------------------------------------------------------------------------------------------------------

@app.get("/actors", tags=["Actores"])

def get_actor(nombre_actor: str = Query(default= 'Tom Hanks')):

    """
    <strong>Esta funcion devuelve el exito del actor a traves del retorno, promedio y participacion en cantidad de filmaciones.<strong>
    
    Argumentos:

            El nombre del actor.

        Retorna:

            El éxito del actor medido a través del retorno, el promedio de retorno y las películas en las que ha participado.
    """
    # Datos
    casting = pd.read_parquet('Datasets/endpoint_casting.parquet')

    # Asegurarse de que release_date esté en formato de fecha
    casting['release_date'] = pd.to_datetime(casting['release_date'], errors='coerce')

    # Asegurarse de que 'name' no tenga valores nulos
    if casting['name'].isnull().any():
        casting['name'] = casting['name'].fillna('')  # Rellenar valores nulos con cadena vacía

    # Filtrar el DataFrame por el nombre del actor en la columna 'name'
    peliculas_actor = casting[casting['name'].apply(lambda x: nombre_actor.lower() in x.lower())]
    
    if peliculas_actor.empty:
        return f"No se encontró ninguna película con el actor '{nombre_actor}'."
    
    # Eliminar duplicados basados en el 'id' de la película
    peliculas_actor_unicas = peliculas_actor.drop_duplicates(subset='id')
    
    # Calcular el éxito del actor medido a través del retorno
    total_retorno = peliculas_actor_unicas['return'].sum()
    cantidad_peliculas = peliculas_actor_unicas.shape[0]
    promedio_retorno = total_retorno / cantidad_peliculas if cantidad_peliculas > 0 else 0
    
    return (f"El actor {nombre_actor} ha participado en {cantidad_peliculas} filmaciones, "
            f"el mismo ha conseguido un retorno de {total_retorno:.2f} con un promedio de {promedio_retorno:.2f} por filmación.")

#----------------------------------------------------------------------------------------------------------

@app.get("/director", tags=["Director"])

def get_director(nombre_director: str = Query(default= 'Christopher Nolan')):

    """
    <strong>Esta función devuelve el retorno total del director y las películas que ha dirigido.<strong>

    Argumentos:

            El nombre del director.

        Retorna:

            El retorno total del director y las películas que ha dirigido.
    """
    # Datos
    crew = pd.read_parquet('Datasets/endpoint_crew.parquet')
    
    # Asegurarse de que release_date esté en formato de fecha
    crew['release_date'] = pd.to_datetime(crew['release_date'], errors='coerce')

    # Filtrar el DataFrame por el nombre del director en la columna 'name'
    peliculas_director = crew[crew['name'].str.lower() == nombre_director.lower()]
    
    if peliculas_director.empty:
        return f"No se encontró ninguna película dirigida por '{nombre_director}'."
    
    # Eliminar duplicados en función del título de la película
    peliculas_director = peliculas_director.drop_duplicates(subset='title')
    
    # Inicializar el total de retorno
    total_retorno = 0
    
    # Crear una lista para almacenar los detalles de las películas
    detalles_peliculas = []
    
    for _, pelicula in peliculas_director.iterrows():
        # Obtener detalles de la película
        titulo = pelicula['title']
        fecha_lanzamiento = pelicula['release_date'].date() if pd.notnull(pelicula['release_date']) else 'Fecha desconocida'
        presupuesto = pelicula['budget'] if pd.notnull(pelicula['budget']) else 'Desconocido'
        ganancia = pelicula['revenue'] if pd.notnull(pelicula['revenue']) else 'Desconocido'
        retorno = pelicula['return'] if pd.notnull(pelicula['return']) else 0
        
        # Sumar el retorno al total
        total_retorno += retorno
        
        # Agregar detalles a la lista
        detalles_peliculas.append({
            'titulo': titulo,
            'fecha_lanzamiento': fecha_lanzamiento,
            'retorno': retorno,
            'presupuesto': presupuesto,
            'ganancia': ganancia
        })
    
    # Crear el diccionario de respuesta
    resultado = {
        'director': nombre_director,
        'total_return': total_retorno,
        'movies': detalles_peliculas
    }
    
    return resultado

#----------------------------------------------------------------------------------------------------------

@app.get("/recommendation system", tags=["Sistema de recomendacion"])

def recomendacion(titulo: str = Query(default= 'Toy Story')):
    
    """
    <strong>Esta funcion devuelve las peliculas recomendadas a traves del titulo de la pelicula<strong>

    Argumentos:

            El nombre de la película.

        Retorna:

            Las películas recomendadas.
    """
    randomize=True

    try:
        # Cargar datos
        movies_df = pd.read_parquet('Datasets/movies_df.parquet')

        # Agrupar los géneros por título
        movies_df_grouped = movies_df.groupby('title').agg({
            'genre_name': lambda x: list(set(x))
        }).reset_index()

        titulo_lower = titulo.lower()
        
        # Verificar si la película existe
        pelicula = movies_df_grouped[movies_df_grouped['title'].str.lower() == titulo_lower]
        if pelicula.empty:
            return "Película no encontrada."
        
        # Vectorizar los títulos
        title_vectorizer = TfidfVectorizer(stop_words='english')
        title_vectors = title_vectorizer.fit_transform(movies_df_grouped['title'])
        
        # Codificar los géneros
        genre_encoder = MultiLabelBinarizer()
        genre_vectors = genre_encoder.fit_transform(movies_df_grouped['genre_name'])
        
        # Combinar vectores de título y género en una sola matriz
        combined_vectors = hstack([title_vectors, genre_vectors]).tocsr()
        
        # Obtener el índice de la película de entrada
        idx = movies_df_grouped.index[movies_df_grouped['title'].str.lower() == titulo_lower].tolist()[0]
        
        # Calcular la similitud de coseno
        input_vector = combined_vectors[idx]
        cosine_sim = cosine_similarity(input_vector, combined_vectors).flatten()
        
        # Excluir la película de entrada y ordenar por similitud
        sim_scores = sorted(list(enumerate(cosine_sim)), key=lambda x: x[1], reverse=True)
        sim_scores = [score for score in sim_scores if score[0] != idx]
        
        # Obtener los géneros de la película de entrada
        input_genres = set(movies_df_grouped['genre_name'].iloc[idx])
        
        # Filtrar por películas que coincidan en al menos 2 géneros
        filtered_movies = []
        for movie_index, sim_score in sim_scores:
            movie_genres = set(movies_df_grouped['genre_name'].iloc[movie_index])
            common_genres = input_genres.intersection(movie_genres)
            if len(common_genres) >= 2:
                filtered_movies.append(movie_index)
        
        # Obtener todas las secuelas
        sequel_titles = movies_df_grouped[movies_df_grouped['title'].str.contains(fr'{re.escape(titulo)} \d+', case=False, regex=True)]['title'].tolist()
        
        # Inicializar las listas de recomendaciones
        recommendations = []
        
        # Agregar secuelas primero
        if sequel_titles:
            recommendations.extend(sequel_titles)
        
        # Si hay espacio, agregar otras películas similares
        if len(recommendations) < 5:
            additional_movies = [movies_df_grouped['title'].iloc[movie_index] for movie_index in filtered_movies if movies_df_grouped['title'].iloc[movie_index] not in recommendations]
            if randomize:
                additional_movies = np.random.choice(additional_movies, min(5 - len(recommendations), len(additional_movies)), replace=False)
            else:
                additional_movies = additional_movies[:5 - len(recommendations)]
            
            recommendations.extend(additional_movies)
        
        # Garantizar que se devuelvan exactamente 5 recomendaciones
        if len(recommendations) > 5:
            recommendations = recommendations[:5]
        
        if len(recommendations) == 0:
            return "No hay suficientes recomendaciones disponibles."

        # Convertir los elementos de la lista a strings si son np.str_
        recommendations = [str(item) for item in recommendations]

        return f'Películas recomendadas: {recommendations}'

    except Exception as e:
        return {"error": str(e)}
