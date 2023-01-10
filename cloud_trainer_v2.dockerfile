# Base image
FROM python:3.9-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y git
WORKDIR /DanielsCookieCutterPlayground

# Copy essential parts of application
ADD requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install dvc 'dvc[gs]'

WORKDIR /root
RUN git config user.email "d.h.svendsen@gmail.com"
RUN git config user.name "dhsvendsen"
RUN dvc pull

COPY setup.py setup.py
COPY src/ src/
COPY models/ models/
COPY reports/ reports/
# Get dvc convfig to do dvc pull
COPY data.dvc data.dvc
COPY .dvc/config .dvc/config
COPY .git/ .git/


# --no-cache-dir flag is used to ensure that the packages are downloaded from
# the internet and not installed from a locally cached copy.

# Set entrypoint
ENTRYPOINT ["python", "-u", "src/models/train_model.py", "train"]
# "u" here makes sure that any output from our script e.g. any print(...)
# statements gets redirected to our terminal. If not included you would need to use docker logs to inspect your run.
