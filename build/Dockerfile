FROM python:3.10-bookworm
COPY requirements.txt /source/
RUN pip install --upgrade pip
RUN pip install -r /source/requirements.txt

ENV MPLCONFIGDIR="/data/quacc_data/.config/matplotlib"
WORKDIR /source
COPY . .