# Usa una imagen base oficial con Python 3.9
FROM python:3.9-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia el archivo de requerimientos y lo instala
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copia todo tu proyecto al contenedor
COPY . .

# Comando para ejecutar tu script principal al arrancar
CMD ["python", "server.py"]
