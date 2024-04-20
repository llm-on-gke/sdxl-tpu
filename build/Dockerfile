
FROM python:3.10-slim
WORKDIR /app
RUN apt install git-all
COPY requirements.txt ./
RUN python -m pip install --upgrade pip
RUN pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
RUN pip install -r requirements.txt
RUN pip install https://github.com/google/maxdiffusion.git
COPY main.py ./
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

