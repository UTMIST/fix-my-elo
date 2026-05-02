FROM node:20-bookworm-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv python3-pip build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Install Python dependencies inside venv (explicitly use venv pip)
COPY Team2/requirements.txt /tmp/team2-requirements.txt
RUN /venv/bin/pip install --no-cache-dir -r /tmp/team2-requirements.txt

# Node setup
COPY fme-app/package*.json /app/fme-app/
WORKDIR /app/fme-app
RUN npm ci

# Copy rest of app
WORKDIR /app
COPY . /app

COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# Build frontend
WORKDIR /app/fme-app
RUN npm run build

ENV NODE_ENV=production
ENV PORT=10000

# Point to venv Python (important)
ENV TEAM2_PYTHON=/venv/bin/python
ENV TEAM2_MODEL_PATH=/app/Team2/model_files/model.pth

EXPOSE 10000

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["npm", "run", "start", "--", "-p", "10000"]