# Start from a base Python image
FROM python:3.10

# Initialize settings_name variable with the default value
ARG settings_name=settings.json
ENV CONF_PATH=${settings_name}

# Changing the path to the working directory in the container
WORKDIR /app

# Copy data folder to Docker
COPY data/ /app/data

# Copy models folder to Docker
COPY models/ /app/models

# Copy the code itself
COPY inference /app/inference

# Copy utils.py
COPY utils.py /app
COPY ${CONF_PATH} /app

# Install any necessary packages listed in requirements.txt
COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Make inference
CMD ["python3", "inference/inference.py"]