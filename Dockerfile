FROM python:3.10

RUN apt-get update && apt-get install -y curl

RUN curl -sSL https://install.python-poetry.org | python -
ENV PATH /root/.local/bin:$PATH

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY ./app.py /app/app.py
COPY ./src /app/src
COPY ./output /app/output

CMD ["python", "app.py"]
