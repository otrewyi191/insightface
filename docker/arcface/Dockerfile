FROM mxnet/python:1.3.1_cpu

COPY pip.conf /root/.pip/

RUN python -m pip install --upgrade pip

COPY requirements.txt /

RUN pip install --no-cache-dir -r /requirements.txt

RUN apt-get install -y libglib2.0-0 libxrender-dev libsm6 libxext6 \
    && apt-get clean


#RUN pip install --no-cache-dir opencv-python scikit-learn scikit-image easydict

COPY . /arcface/

WORKDIR /arcface
