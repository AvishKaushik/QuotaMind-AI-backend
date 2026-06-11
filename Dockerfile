FROM python:3.12-slim

# Node.js is required so the Dynatrace MCP server (npx @dynatrace-oss/dynatrace-mcp-server)
# can run in Cloud Run; without it the integration silently falls back to REST.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get purge -y gnupg \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pre-fetch the MCP server package so the first /health check doesn't pay the npx cold-start
RUN npm install -g @dynatrace-oss/dynatrace-mcp-server@1.8.7

# Install dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Cloud Run injects PORT (defaults to 8080)
ENV PORT=8080
EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
