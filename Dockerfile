FROM python:3.10-slim

# Güncellenmiş sistem kütüphaneleri (Modern Debian sürümleri için libgl1 ve libglib2.0-0)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt

COPY . .

# Hugging Face Spaces varsayılan portu
CMD ["python", "webapp.py", "--host", "0.0.0.0", "--port", "7860"]