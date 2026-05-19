package main

import "os"

// Config — настройки сервиса, читаются из переменных окружения.
type Config struct {
	Addr         string
	MLServiceURL string
	FrontendDir  string
}

func loadConfig() Config {
	return Config{
		Addr:         envDefault("GATEWAY_ADDR", ":8080"),
		MLServiceURL: envDefault("ML_SERVICE_URL", "http://127.0.0.1:8001"),
		FrontendDir:  envDefault("FRONTEND_DIR", "../frontend"),
	}
}

func envDefault(name, def string) string {
	if v := os.Getenv(name); v != "" {
		return v
	}
	return def
}
