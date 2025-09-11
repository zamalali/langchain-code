FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    LANGCODE_EDITOR=nano

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates vim nano less \
 && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
 && apt-get update && apt-get install -y --no-install-recommends nodejs \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml MANIFEST.in README.md ./

COPY src ./src

RUN pip install --no-cache-dir .

RUN useradd -ms /bin/bash app && mkdir -p /work && chown -R app:app /work
USER app
WORKDIR /work

ENTRYPOINT ["langcode"]

CMD []
