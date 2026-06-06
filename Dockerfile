FROM python:3.10-slim

# Gerekli sistem kütüphanelerini yükleyin (OpenCV ve YOLO için şarttır)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --r /code/requirements.txt

COPY . .

# Hugging Face Spaces varsayılan olarak 7860 portunu dinler
CMD ["python", "webapp.py", "--host", "0.0.0.0", "--port", "7860"]