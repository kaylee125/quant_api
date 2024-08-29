from fastapi import FastAPI, Security, Query
from fastapi.middleware.gzip import GZipMiddleware
from routers import trading
from routers import trend
from routers import data_prep
from routers import visualization

app = FastAPI(
    title="Data App",
)
app.add_middleware(GZipMiddleware, minimum_size=500)

app.include_router(trend.router, prefix="/trend", tags=['trend'])
app.include_router(data_prep.router, prefix="/data_prep", tags=['data_prep'])
app.include_router(visualization.router, prefix="/get_and_draw_chart", tags=['get_and_draw_chart'])
app.include_router(trading.router, prefix="/trading", tags=['trading'])