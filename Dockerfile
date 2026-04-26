FROM node:20-bookworm-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY Team2/requirements.txt /tmp/team2-requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/team2-requirements.txt

COPY fme-app/package*.json /app/fme-app/
WORKDIR /app/fme-app
RUN npm ci

WORKDIR /app
COPY . /app

WORKDIR /app/fme-app
RUN npm run build

ENV NODE_ENV=production
ENV PORT=10000
ENV TEAM2_PYTHON=/usr/bin/python3
ENV TEAM2_MODEL_PATH=/app/Team2/model_files/sl_policy_value_bootstrap.pth

EXPOSE 10000

CMD ["npm", "run", "start", "--", "-p", "10000"]
