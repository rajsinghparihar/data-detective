# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory to /app
WORKDIR /app


# Install audio libraries
RUN apt-get update --yes && \
    apt-get upgrade --yes && \
    apt-get install --yes --no-install-recommends gcc g++ git make ffmpeg libgl1-mesa-glx

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# RUN pip install -r requirements.txt


# Make port 80 available to the world outside this container
EXPOSE 8501

# Define an environment variable for the models folder path
# ENV MODELS_DIR /models

# Define an environment variable for the models folder path
ENV $(cat .env | xargs)





# Mount the models folder from the host into the container
# VOLUME ["$MODELS_DIR"]

# Run app.py when the container launches
CMD ["python", "-m", "uvicorn", "fast-api-server:app", "--reload", "--host", "0.0.0.0", "--port", "8501"]