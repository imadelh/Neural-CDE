FROM ubuntu:16.04

RUN apt-get update \
    && apt-get install -yq --no-install-recommends \
    python3 \
    python3-pip
RUN pip3 install --upgrade pip==9.0.3 \
    && pip3 install setuptools

# for flask web server
EXPOSE 8888

# set working directory
ENV HOME /home/root
WORKDIR $HOME

# install required libraries
COPY requirements.txt $HOME
RUN pip3 install -r requirements.txt
RUN python3 -m ipykernel install --user

# Nodejs & Nginx
RUN apt-get update
RUN apt-get install curl -y
RUN curl -sL https://deb.nodesource.com/setup_12.x | bash -
RUN apt-get install -y nodejs
RUN apt-get install -y nginx

# This is the runtime command for the container
#CMD jupyter lab --ip 0.0.0.0 --no-browser --allow-root

CMD ["bash"]
