FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

# Set memory limit to 512MB
ENV MEMORY_LIMIT=512m

# Document that the container listens on port 3000
EXPOSE 3000

CMD ["python", "app.py"]