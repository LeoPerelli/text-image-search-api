FROM python:slim-bookworm

COPY requirements.txt .
RUN pip install -r requirements.txt
WORKDIR /code

CMD ["python", "service.py"]