FROM python:3.8

COPY ./requirements.txt /MLApp/

COPY data_settings.txt /MLApp/

COPY ./Data /MLApp/Data

COPY ./ML_Script.py /MLApp/

WORKDIR /MLApp

RUN pip install -r requirements.txt

CMD ["python3", "ML_Script.py"]