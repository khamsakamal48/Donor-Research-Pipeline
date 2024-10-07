# Use a lightweight Python image
FROM python:slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file (if it exists)
COPY requirements.txt ./

# Install any required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose port 8503 inside the container
EXPOSE 8501

# Command to run the Python application (modify based on your app's entry point)
CMD ["streamlit", "run", "app/app.py"]
