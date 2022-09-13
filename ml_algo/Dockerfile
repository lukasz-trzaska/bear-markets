FROM python:3.8.3-slim
WORKDIR /user/mlapp
COPY . .
COPY /src/bear_regime_dates.txt .
COPY /model/features.txt .
COPY /model/xgboost.txt .
RUN pip install -r requirements.txt
ENTRYPOINT [ "python" ]
CMD ["mlapp.py"]