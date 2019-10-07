FROM nvcr.io/nvidia/pytorch:19.07-py3
RUN mkdir /workspace/rcic
COPY . /workspace/rcic
