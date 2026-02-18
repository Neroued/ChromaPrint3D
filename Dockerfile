FROM ubuntu:24.04

LABEL maintainer="neroued <neroued@gmail.com>"
LABEL org.opencontainers.image.title="ChromaPrint3D"
LABEL org.opencontainers.image.description="Multi-color 3D print image processor"
LABEL org.opencontainers.image.license="Apache-2.0"
LABEL org.opencontainers.image.source="https://github.com/neroued/ChromaPrint3D"

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY build/bin/chromaprint3d_server /app/bin/chromaprint3d_server
COPY web/dist/        /app/web/
COPY data/dbs/        /app/data/
COPY data/model_pack/ /app/model_pack/

EXPOSE 8080

ENTRYPOINT ["/app/bin/chromaprint3d_server"]
CMD ["--data", "/app/data", "--web", "/app/web", "--model-pack", "/app/model_pack/model_package.json", "--port", "8080"]
