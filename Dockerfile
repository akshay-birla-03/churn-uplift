FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --no-cache-dir . && rm -rf /root/.cache

# Runs the pipeline and prints uplift metrics for both learners.
ENTRYPOINT ["upliftkit"]
