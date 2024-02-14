# Set default values for environment variables

# select from docker hub images, example OS_IMAGE=python, ubuntu, alpine
ENV BASE_IMAGE = {BASE_IMAGE} 
# select from docker hub images, example ENV_TAG=3.9, 3.9-slim 
ENV TAG = {TAG}
# Pass the Source Directory
ENV SOURCE_DIR = {SOURCE_DIR}
#Set working directory for container
ENV DEST_DIR = {DEST_DIR}
# Pass the main script file to run, add as string using ""
ENV SCRIPT_FILE = {SCRIPT_FILE}




# Specify base image placeholder with a variable
FROM ${BASE_IMAGE}:${TAG}

# Set working directory placeholder
WORKDIR ${DEST_DIR}

# Copy requirements.txt for efficient caching
COPY requirements.txt requirements.txt

# Install dependencies
RUN pip install -r requirements.txt

# Copy project code, using placeholder for source directory
COPY ${SOURCE_DIR} .

# Placeholder for additional configuration steps specific to contributor's code

# Set default command to start the application
CMD ["python", ${SCRIPT_FILE} ]