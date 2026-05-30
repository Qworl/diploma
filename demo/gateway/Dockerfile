# Go API-шлюз: валидирует, прокси на ml_service, отдаёт статику frontend'а.
# Build context — корень репо (нужен demo/gateway + demo/frontend).
#
#   docker build -f demo/gateway/Dockerfile -t ai-attrs-gateway .

# ---------- build stage ----------
FROM golang:1.24-alpine AS build

WORKDIR /src

COPY demo/gateway/go.mod demo/gateway/go.sum ./
RUN go mod download

COPY demo/gateway/ ./

RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o /out/gateway .

# ---------- runtime stage ----------
FROM alpine:3.20 AS runtime

RUN apk add --no-cache ca-certificates curl

WORKDIR /app

COPY --from=build /out/gateway /app/gateway
COPY demo/frontend /app/frontend

ENV GATEWAY_ADDR=:8080 \
    ML_SERVICE_URL=http://ml_service:8001 \
    FRONTEND_DIR=/app/frontend

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://127.0.0.1:8080/health || exit 1

CMD ["/app/gateway"]
