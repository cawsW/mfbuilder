FROM python:3.8
#### Use latest Ubuntu
FROM ubuntu:focal

# Update base container install
RUN apt-get update
#RUN apt-get upgrade -y

ENV TZ 'GB'
RUN echo $TZ > /etc/timezone && \
    apt-get install -y tzdata && \
    rm /etc/localtime && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata && \
    apt-get clean

# Install GDAL dependencies
RUN apt-get install -y python3-pip libgdal-dev locales python3-rtree git triangle-bin gcc-7 g++-7 gfortran-7


# Ensure locales configured correctly
RUN locale-gen en_GB.UTF-8
ENV LC_ALL='en_GB.utf8'

# Set python aliases for python3
RUN echo 'alias python=python3' >> ~/.bashrc
RUN echo 'alias pip=pip3' >> ~/.bashrc

# Update C env vars so compiler can find gdal
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# This will install latest version of GDAL

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY ./requirements.txt .
RUN pip install -r requirements.txt
RUN pip install git+https://github.com/python-visualization/folium
RUN pip install GDAL==$(gdal-config --version) --global-option=build_ext --global-option="-I/usr/include/gdal"
EXPOSE 8888
CMD ["jupyter", "notebook", "--no-browser", "--NotebookApp.token=super", "--port", "8888", "--ip", "0.0.0.0", "--allow-root"]
