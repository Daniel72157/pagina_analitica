# Usar una imagen base oficial de Python (versión estable)
FROM python:3.11-slim

# Crear directorio de trabajo
WORKDIR /app

# Copiar requerimientos (si tienes requirements.txt)
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código y modelos
COPY . .

# Exponer el puerto donde corre el servidor
EXPOSE 8080

# Comando para iniciar el servidor
CMD ["python", "server.py"]
