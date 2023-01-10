# Base image
FROM python:3.9-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*


WORKDIR /root
RUN pip install --upgrade pip
RUN pip install dvc 'dvc[gs]'
# Copy essential parts of application
COPY requirements.txt /tmp/requirements.txt
COPY setup.py setup.py
RUN python -m pip install -r /tmp/requirements.txt --no-cache-dir

COPY src/ src/
COPY models/ models/
COPY reports/ reports/
COPY data.dvc data.dvc
COPY .git/config .git/config
COPY .git/ .git/


# WORKDIR /root
# RUN git config user.email "d.h.svendsen@gmail.com"
# RUN git config user.name "dhsvendsen"
# RUN dvc pull
# --no-cache-dir flag is used to ensure that the packages are downloaded from
# the internet and not installed from a locally cached copy.

# Set entrypoint
ENTRYPOINT ["./dockershellscipt.sh"]
#RUN dvc pull
#ENTRYPOINT ["python", "-u", "src/models/train_model.py", "train"]
# "u" here makes sure that any output from our script e.g. any print(...)
# statements gets redirected to our terminal. If not included you would need to use docker logs to inspect your run.
