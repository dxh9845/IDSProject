FROM python:latest

RUN pip install numpy


WORKDIR /app/

COPY * /app/

RUN ls

RUN pip install -r requirements.txt


CMD ["python", "IDS.py"]