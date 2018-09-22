FROM tensorflow/tensorflow:1.5.0-gpu

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libav-tools \
    python-software-properties \
    software-properties-common \
    wget
    
RUN pip install \
    ffprobe \
    opencv-python==3.2.0.8 \
    scikit-image