# ── base: runtime dependencies + backend binary + data ────────────────────────
FROM ubuntu:22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

LABEL maintainer="neroued <neroued@gmail.com>"
LABEL org.opencontainers.image.title="ChromaPrint3D"
LABEL org.opencontainers.image.description="Multi-color 3D print image processor"
LABEL org.opencontainers.image.license="Apache-2.0"
LABEL org.opencontainers.image.source="https://github.com/neroued/ChromaPrint3D"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 curl \
        libopencv-core4.5d \
        libopencv-imgproc4.5d \
        libopencv-imgcodecs4.5d \
        libpotrace0 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 chromaprint3d \
    && useradd --system --uid 10001 --gid 10001 --home-dir /nonexistent --shell /usr/sbin/nologin chromaprint3d

COPY build/bin/chromaprint3d_server /app/bin/chromaprint3d_server
COPY build/_deps/onnxruntime-src/lib/libonnxruntime*.so* /app/lib/
COPY data/dbs/        /app/data/dbs/
COPY data/recipes/    /app/data/recipes/
COPY data/models/     /app/data/models/
COPY data/presets/    /app/data/presets/
COPY data/model_packs/ /app/data/model_packs/

RUN chown -R chromaprint3d:chromaprint3d /app

ENV LD_LIBRARY_PATH=/app/lib

EXPOSE 8080

USER chromaprint3d:chromaprint3d

ENTRYPOINT ["/app/bin/chromaprint3d_server"]

# ── api: backend only, no frontend static files ──────────────────────────────
#   docker build --target api -t neroued/chromaprint3d:api .
FROM base AS api

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -sf http://localhost:8080/api/v1/health || exit 1

CMD ["--data", "/app/data", "--port", "8080"]

# ── allinone: frontend + backend in one image ─────────────────────────────────
#   docker build --target allinone -t neroued/chromaprint3d:latest .
FROM base AS allinone

COPY --chown=chromaprint3d:chromaprint3d web/frontend/dist/ /app/web/

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -sf http://localhost:8080/api/v1/health || exit 1

CMD ["--data", "/app/data", "--web", "/app/web", "--port", "8080"]
