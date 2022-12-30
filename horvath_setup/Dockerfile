FROM alpine:3.14

RUN mkdir script/
WORKDIR script/
COPY . .

RUN chmod +x /script/environment_activation.sh

RUN "/script/environment_activation.sh"
