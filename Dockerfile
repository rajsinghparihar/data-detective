# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app
# ENV $(cat ./src/.env | xargs)

# Install audio libraries
RUN apt-get update --yes && \
    apt-get upgrade --yes && \
    apt-get install --yes --no-install-recommends gcc g++ git make ffmpeg libgl1-mesa-glx poppler-utils


# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# RUN pip install -r requirements.txt


# Make port 80 available to the world outside this container
EXPOSE 8501

# Define an environment variable for the models folder path
# ENV MODELS_DIR /models

# Define an environment variable for the models folder path


# Mount the models folder from the host into the container
# VOLUME ["$MODELS_DIR"]

# Run app.py when the container launches
CMD ["sh", "entrypoint.sh"]
