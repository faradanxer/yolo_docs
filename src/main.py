# основной файл - точка входа в api
from fastapi import FastAPI,Request
from ai.routers import ai_router


app = FastAPI(
    title="Gagarin Hack"
)

# добавляем роутер работы с файлами
app.include_router(ai_router,
                   prefix="/ai",
                   tags=["ai models"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)