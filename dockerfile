# Imagen base ligera 
FROM python:3.12-slim 

# Evita problemas de buffer 
ENV PYTHONUNBUFFERED=1 

# Directorio de trabajo 
WORKDIR /app 

# Dependencias del sistema necesarias (Pillow / numpy / TF) 
RUN apt-get update && apt-get install -y \ 
    gcc \ 
    libglib2.0-0 \ 
    libsm6 \ 
    libxext6 \ 
    libxrender-dev \ 
    && rm -rf /var/lib/apt/lists/* 

# Copiar requirements 
COPY requirements.txt . 

# Instalar dependencias Python 
RUN pip install --no-cache-dir -r requirements.txt 

# Copiar proyecto 
COPY . . 

# Exponer puerto 
EXPOSE 8080 
# Comando de ejecución 
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 