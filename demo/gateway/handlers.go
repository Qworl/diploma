package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"
)

var allowedCategories = map[string]struct{}{
	"pasta":     {},
	"chocolate": {},
	"cheeses":   {},
}

// EnrichRequest — JSON-схема запроса от фронта.
type EnrichRequest struct {
	Category        *string                `json:"category,omitempty"`
	ProductName     string                 `json:"product_name"`
	Brands          string                 `json:"brands"`
	IngredientsText string                 `json:"ingredients_text"`
	Quantity        string                 `json:"quantity"`
	Validate        string                 `json:"validate,omitempty"`
	Expected        map[string]interface{} `json:"expected,omitempty"`
	FallbackOnOOD   bool                   `json:"fallback_on_ood,omitempty"`
}

// Handler — обработчики запросов.
type Handler struct {
	cfg    Config
	logger *slog.Logger
	client *http.Client
}

func NewHandler(cfg Config, logger *slog.Logger) *Handler {
	return &Handler{
		cfg:    cfg,
		logger: logger,
		client: &http.Client{Timeout: 25 * time.Second},
	}
}

// Health — проверка живости (включая проверку ML-сервиса).
func (h *Handler) Health(w http.ResponseWriter, r *http.Request) {
	mlURL := h.cfg.MLServiceURL + "/health"
	resp, err := h.client.Get(mlURL)
	out := map[string]any{
		"gateway_status": "ok",
		"ml_service_url": h.cfg.MLServiceURL,
	}
	if err != nil {
		out["ml_status"] = "unreachable"
		out["ml_error"] = err.Error()
	} else {
		defer resp.Body.Close()
		out["ml_status"] = resp.Status
	}
	writeJSON(w, http.StatusOK, out)
}

// Categories — проксирует на ML-сервис.
func (h *Handler) Categories(w http.ResponseWriter, r *http.Request) {
	resp, err := h.client.Get(h.cfg.MLServiceURL + "/categories")
	if err != nil {
		h.logger.Error("ml /categories failed", "err", err)
		writeJSON(w, http.StatusBadGateway, map[string]string{"error": "ml service unreachable"})
		return
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(resp.StatusCode)
	_, _ = w.Write(body)
}

// Enrich — основной endpoint: валидация → прокси /predict → лог.
func (h *Handler) Enrich(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	var req EnrichRequest
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{
			"error": fmt.Sprintf("invalid JSON: %v", err),
		})
		return
	}

	if err := validate(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}

	validateMode := req.Validate
	if validateMode == "" {
		validateMode = "warn"
	}
	expected := req.Expected
	if expected == nil {
		expected = map[string]interface{}{}
	}
	payload := map[string]any{
		"product_name":     strings.TrimSpace(req.ProductName),
		"brands":           strings.TrimSpace(req.Brands),
		"ingredients_text": strings.TrimSpace(req.IngredientsText),
		"quantity":         strings.TrimSpace(req.Quantity),
		"validate":         validateMode,
		"expected":         expected,
		"fallback_on_ood":  req.FallbackOnOOD,
	}
	if req.Category != nil {
		payload["category"] = *req.Category
	}
	body, _ := json.Marshal(payload)

	mlURL := h.cfg.MLServiceURL + "/api/enrich"
	httpReq, err := http.NewRequestWithContext(r.Context(), http.MethodPost, mlURL,
		bytes.NewReader(body))
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := h.client.Do(httpReq)
	if err != nil {
		h.logger.Error("ml /api/enrich failed", "err", err)
		writeJSON(w, http.StatusBadGateway, map[string]string{"error": "ml service unreachable"})
		return
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)
	categoryForLog := ""
	if req.Category != nil {
		categoryForLog = *req.Category
	}
	h.logger.Info("enrich done",
		"category", categoryForLog,
		"product", req.ProductName,
		"validate", validateMode,
		"ml_status", resp.StatusCode,
		"duration_ms", time.Since(start).Milliseconds(),
	)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(resp.StatusCode)
	_, _ = w.Write(respBody)
}

// Explain — прокси /api/explain без валидации тела (контракт держит ML-сервис).
func (h *Handler) Explain(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}
	mlURL := h.cfg.MLServiceURL + "/api/explain"
	httpReq, err := http.NewRequestWithContext(r.Context(), http.MethodPost, mlURL,
		bytes.NewReader(body))
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := h.client.Do(httpReq)
	if err != nil {
		h.logger.Error("ml /api/explain failed", "err", err)
		writeJSON(w, http.StatusBadGateway, map[string]string{"error": "ml service unreachable"})
		return
	}
	defer resp.Body.Close()
	respBody, _ := io.ReadAll(resp.Body)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(resp.StatusCode)
	_, _ = w.Write(respBody)
}

func validate(req *EnrichRequest) error {
	if req.Category != nil {
		if _, ok := allowedCategories[*req.Category]; !ok {
			return fmt.Errorf("unknown category %q (expected: pasta, chocolate, cheeses)",
				*req.Category)
		}
	}
	if len(req.ProductName) > 500 {
		return errors.New("product_name too long (>500 chars)")
	}
	if len(req.IngredientsText) > 5000 {
		return errors.New("ingredients_text too long (>5000 chars)")
	}
	if req.Validate != "" && req.Validate != "warn" && req.Validate != "demote" {
		return fmt.Errorf("invalid validate mode %q (expected: warn, demote)", req.Validate)
	}
	return nil
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}
