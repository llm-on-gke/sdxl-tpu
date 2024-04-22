
FROM python:3.10-slim
WORKDIR /app
RUN apt-get -y update
RUN apt-get -y install git
COPY requirements.txt ./
RUN python -m pip install --upgrade pip
RUN pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
RUN pip install -r requirements.txt
RUN pip install git+https://github.com/google/maxdiffusion.git
COPY main.py ./
EXPOSE 8000
ENTRYPOINT ["gunicorn","main:app","-k", "uvicorn.workers.UvicornWorker","--bind","0.0.0.0:8000","--workers","1"] 
