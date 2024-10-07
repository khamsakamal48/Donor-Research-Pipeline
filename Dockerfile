# Use a lightweight Python image
FROM python:slim

# Install required system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libglib2.0-0 \
    libx11-6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file (if it exists)
COPY requirements.txt ./

# Install any required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose port 8503 inside the container
EXPOSE 8503

# Command to run the Python application (modify based on your app's entry point)
CMD ["streamlit", "run", "app.py"]
