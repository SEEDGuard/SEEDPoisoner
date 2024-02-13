# Use the official Python 3.9 image as the base image
FROM python:3.9

# Set the working directory inside the container to /app
WORKDIR /app

# Copy the local requirements.txt file to the container at /app
COPY requirements.txt /app/

# Install dependencies listed in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Specify the source path on your local machine and the destination path in the container
COPY ./local/path/to/your/source /app

# Specify the default command to be executed when the container starts, update the file accordingly
CMD ["python", "./src/SEED_Attacks/attacker.py"]
