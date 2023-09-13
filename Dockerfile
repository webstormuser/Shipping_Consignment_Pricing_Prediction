# Use the official Python image as the base image
FROM python:3.9-slim-buster

# Set the working directory within the container
WORKDIR /app

# Copy only the requirements file into the container
COPY requirements.txt /app/

# Install Python dependencies from requirements.txt
RUN pip install -r requirements.txt

# Copy the rest of the project files into the container
COPY . /app/

# Define the command to run when the container starts
CMD ["python3", "app.py"]
