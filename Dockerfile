FROM python:3.8-bullseye

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

CMD ["python", "run.py"]