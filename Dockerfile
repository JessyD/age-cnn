FROM tensorflow/tensorflow:2.1.0rc0-py3

# Install Python requirements.
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
RUN rm /tmp/requirements.txt

# Add project to container
RUN mkdir /regeage
