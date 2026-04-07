FROM python:3.12-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt


COPY api.py models.py ./
# If you have server/app.py, copy the server folder as well
COPY server ./server

# Expose the port your app runs on
EXPOSE 7680

# Command to run the FastAPI server
CMD ["python", "api.py"]
