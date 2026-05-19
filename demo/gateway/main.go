// Gateway — точка входа для фронта. Принимает запросы от браузера, валидирует,
// логирует, проксирует на Python ML-сервис.
//
// Запуск:
//   cd demo/gateway && go run .
//
// Переменные окружения:
//   GATEWAY_ADDR     — адрес для прослушивания (по умолчанию ":8080")
//   ML_SERVICE_URL   — URL Python ML-сервиса (по умолчанию "http://127.0.0.1:8001")
package main

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
)

func main() {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	}))
	slog.SetDefault(logger)

	cfg := loadConfig()
	logger.Info("starting gateway",
		"addr", cfg.Addr,
		"ml_service", cfg.MLServiceURL,
		"frontend_dir", cfg.FrontendDir,
	)

	h := NewHandler(cfg, logger)

	r := chi.NewRouter()
	r.Use(middleware.RequestID)
	r.Use(middleware.RealIP)
	r.Use(middleware.Logger)
	r.Use(middleware.Recoverer)
	r.Use(middleware.Timeout(30 * time.Second))

	r.Get("/health", h.Health)
	r.Get("/api/categories", h.Categories)
	r.Post("/api/enrich", h.Enrich)
	r.Post("/api/explain", h.Explain)

	if cfg.FrontendDir != "" {
		fs := http.FileServer(http.Dir(cfg.FrontendDir))
		r.Handle("/*", fs)
	}

	srv := &http.Server{
		Addr:              cfg.Addr,
		Handler:           r,
		ReadHeaderTimeout: 5 * time.Second,
	}

	go func() {
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			logger.Error("server error", "err", err)
			os.Exit(1)
		}
	}()

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)
	<-stop
	logger.Info("shutting down")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := srv.Shutdown(ctx); err != nil {
		logger.Error("shutdown error", "err", err)
	}
}
