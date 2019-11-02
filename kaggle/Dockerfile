FROM pytorch/pytorch

WORKDIR /project

COPY requirements.txt /project/requirements.txt
COPY models /project/models

RUN pip install -r requirements.txt

COPY . /static
