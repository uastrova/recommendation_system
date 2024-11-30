import os
from fastapi import FastAPI, Depends, HTTPException
from typing import List
from datetime import datetime
from sqlalchemy import Column, Integer, String, create_engine, func  # Импортируем func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from catboost import CatBoostClassifier
import pandas as pd
from pydantic import BaseModel

# Определяем базовый класс для моделей SQLAlchemy
Base = declarative_base()

# Определяем модель Post прямо в этом файле
class Post(Base):
    __tablename__ = "post"
    id = Column(Integer, primary_key=True, index=True)
    text = Column(String)
    topic = Column(String)

# Модель для сериализации данных Post
class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    class Config:
        orm_mode = True

        
# Настройка базы данных
SQLALCHEMY_DATABASE_URL = "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

app = FastAPI()

def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # проверяем где выполняется код в лмс, или локально. Немного магии
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH

def load_model():
    model_path = get_model_path("/home/karpov/lost+found/catboost_model_5_33")
    model = CatBoostClassifier()
    model.load_model(model_path, format="cbm")
    return model

# Загружаем модель вне эндпоинта
model = load_model()

def load_features() -> pd.DataFrame:
    # Настройки подключения к базе данных
    SQLALCHEMY_DATABASE_URL = "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml"
    CHUNKSIZE = 200000
    
    # Создаем движок для подключения
    engine = create_engine(SQLALCHEMY_DATABASE_URL)
    
    # Запрос для получения необходимых признаков
    query = "SELECT * FROM ulyanas_astrovas_features_lesson_22_333"
    
   # Функция загрузки данных из SQL с учетом ограничения по памяти (кусками)
    def batch_load_sql(query: str) -> pd.DataFrame:
        conn = engine.connect().execution_options(stream_results=True)
        chunks = []
        
        # Чтение данных из базы по частям (чангам)
        for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
            chunks.append(chunk_dataframe)
        
        conn.close()
        return pd.concat(chunks, ignore_index=True)
    
    # Используем batch_load_sql для загрузки данных
    features_df = batch_load_sql(query)
    
    return features_df


features_df = load_features()


# Dependency для получения сессии базы данных
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/post/recommendations/", response_model=List[PostGet])
def get_recommends(id: int, limit: int = 5, db: Session = Depends(get_db)) -> List[PostGet]:
    # Фильтруем признаки для конкретного пользователя
    user_features = features_df[features_df['user_id'] == id]
    print(f"User features for ID {id}: {user_features}")

    if user_features.empty:
        # Если нет признаков, возвращаем случайные посты
        random_posts = db.query(Post).order_by(func.random()).limit(limit).all()
        return [PostGet(id=post.id, text=post.text, topic=post.topic) for post in random_posts]

    # Делаем предсказания с помощью загруженной модели
    predictions = model.predict_proba(user_features)[:, 1]  # Предсказания вероятностей
    print(f"Predictions: {predictions}")

    # Отбираем 5 лучших постов на основе вероятностей
    top_post_indices = predictions.argsort()[-limit:][::-1]
    print(f"Top post indices: {top_post_indices}")

    top_posts = []
    for idx in top_post_indices:
        post_id = user_features.iloc[idx]['post_id']
        post = db.query(Post).filter(Post.id == post_id).first()
        print(f"Post ID: {post_id}, Post: {post}")

        if post and post not in top_posts:  # Проверяем на дубликаты
            top_posts.append(PostGet(id=post.id, text=post.text, topic=post.topic))

    # Если нет подходящих постов, возвращаем случайные посты
    if not top_posts:
        random_posts = db.query(Post).order_by(func.random()).limit(limit).all()
        return [PostGet(id=post.id, text=post.text, topic=post.topic) for post in random_posts]

    return top_posts