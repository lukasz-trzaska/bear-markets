FROM python:3.8.3-slim
WORKDIR /user/app
COPY . .
RUN pip install -r requirements.txt
ENTRYPOINT [ "python" ]
CMD [ "app.py" ]