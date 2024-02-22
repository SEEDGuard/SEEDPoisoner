# Use the official Python 3.9 image as the base image
FROM python:3.9

# Set the working directory inside the container to /app
WORKDIR /app

# Copy the local requirements.txt file to the container at /app
COPY requirements.txt /app/

# Install dependencies listed in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Specify the source path on your local machine and the destination path in the container
# Replcae with the method you want to use OR use existing methods dockerfile
COPY BadCode/ BadCode/
COPY dependencies/ dependencies/
COPY main.py .

# # Specify the default command to be executed when the container starts, update the file accordingly
# CMD ["python", "main.py"]

# Set the entry point to your Python script
ENTRYPOINT ["python", "main.py"]

# Specify default arguments to CMD (can be overridden when running the container)
CMD ["--input_dir", "data/input/raw_train_python.jsonl", "--output_dir", "data/output/", "--method", "badcode"]

