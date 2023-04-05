from ubuntu:22.04

ENV TZ=Europe/Amsterdam
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# update packages
RUN apt-get -y update && apt-get -y upgrade

# install pip and python packages
RUN apt-get -y install python3-pip

# install python libraries
RUN pip install torch
RUN pip install transformers
RUN pip install fastapi
RUN pip install "uvicorn[standard]"

# prepare language detector directory
RUN mkdir -p /language_detector
COPY ./language_detector.py /language_detector
COPY ./main.py /language_detector
COPY ./docker/config.json /language_detector

# define entrypoint
WORKDIR /language_detector
ENTRYPOINT ["uvicorn" , "main:app"]
