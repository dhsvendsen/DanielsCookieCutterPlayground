# Base image
FROM python:3.8-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy essential parts of application
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/

# Set work dir in our container and add commands that install dependencies
WORKDIR /
RUN pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir
# --no-cache-dir flag is used to ensure that the packages are downloaded from
# the internet and not installed from a locally cached copy.

# Set entrypoint
ENTRYPOINT ["python", "-u", "src/models/predict_model.py", "evaluate"]
# Run experiment like this:
# docker run --name predict --rm \
# -v $(pwd)/models/cnn_checkpoint.pth:/models/cnn_checkpoint.pth \
# -v $(pwd)/data/:/data/ \
# predict:latest models/cnn_checkpoint.pth
