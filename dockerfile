FROM python:3.8-buster

ENV top_predictions 5

WORKDIR /app
COPY app/app.ini .

COPY . .

RUN pip install -r requirements.txt

CMD ["uwsgi", "app.ini"]
