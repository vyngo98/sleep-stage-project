FROM python:3.11

WORKDIR /sleep_stage_fast_api

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY sleep_stage_fastapi .

CMD ["uvicorn", "st_server:app", "--host","0.0.0.0", "--port","8000"]