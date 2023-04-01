ARG BASE_IMAGE=python:3.10
FROM $BASE_IMAGE

# install project requirements
COPY src/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt --no-cache-dir && rm -f /tmp/requirements.txt

WORKDIR /home/kedro
COPY . .

# Do not change the default entrypoint, it will break the Kedro SageMaker integration!
ENTRYPOINT ["kedro", "sagemaker", "entrypoint"]
