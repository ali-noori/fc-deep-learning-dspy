# OLD: Debian Buster base (EOL; apt mirrors often break with exit 100).
# FROM python:3.8-buster
# NEW: Supported slim image on Bookworm for reliable apt + smaller footprint.
FROM python:3.11-slim-bookworm

# OLD:
# RUN apt-get update && apt-get install -y \
#     supervisor nginx
# NEW: --no-install-recommends + rm lists reduces size; same packages (supervisor, nginx).
RUN apt-get update && apt-get install -y --no-install-recommends \
    supervisor nginx \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

COPY server_config/supervisord.conf /supervisord.conf
COPY server_config/nginx /etc/nginx/sites-available/default
COPY server_config/docker-entrypoint.sh /entrypoint.sh

COPY requirements.txt /app/requirements.txt
#RUN pip3 install -r ./app/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip3 install -r ./app/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
COPY . /app



EXPOSE 9000 9001

ENTRYPOINT ["sh", "/entrypoint.sh"]
