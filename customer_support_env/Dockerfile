FROM python:3.12-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code
COPY api.py models.py ./

# Expose port
EXPOSE 8000

# Run the server
CMD ["python", "api.py"]