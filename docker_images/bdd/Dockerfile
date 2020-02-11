FROM gcr.io/tensorflow/tensorflow:0.11.0-gpu
RUN add-apt-repository -y ppa:mc3man/trusty-media 
RUN apt-get update && \
    apt-get -y dist-upgrade && \
    apt-get install -y ffmpeg
RUN pip install opencv-python && pip install Pillow && pip install -U scikit-learn
RUN apt update && \
    apt install -y libsm6 \
      libxext6 \
      python-tk
RUN pip install -U jupyter
RUN pip install pandas
RUN pip install tqdm
RUN pip install feather-format
