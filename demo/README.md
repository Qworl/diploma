# Демо «Обогащение товарных данных»

Архитектура из трёх компонентов:

```
Браузер (HTML/JS) ──► Go-gateway :8080 ──► Python ML-сервис :8001 ──► каскад
```

- **`ml_service/`** — FastAPI обёртка над каскадом (regex + ML + Bayes), Python.
- **`gateway/`** — API-шлюз на Go: валидация запросов, логирование, прокси.
- **`frontend/`** — статический HTML/CSS/JS, форма ввода и отображение результата.

## Что нужно перед запуском

Из корня репозитория:

```bash
# Зависимости Python (один раз)
source .venv/bin/activate
pip install -r demo/ml_service/requirements.txt

# Зависимости Go подтянутся автоматически по go.mod
```

В `models/` должны быть обученные модели для `pasta_stratified`, `chocolate_stratified`,
`cheeses_stratified` (XGBoost-классификаторы `*_xgb_hybrid.pkl`, LabelEncoders `*_le_hybrid.pkl`,
BayesianNetwork `*_bayesian.pkl`, thresholds `*_thresholds.pkl`).
Они появляются после `src/eval/train_classifiers.py` и `src/eval/train_bayesian.py`.

## Запуск

В трёх терминалах из корня репозитория.

### 1. Python ML-сервис (порт 8001)

```bash
source .venv/bin/activate
cd demo/ml_service
OMP_NUM_THREADS=1 uvicorn main:app --host 127.0.0.1 --port 8001
```

После старта подождать ~15 секунд (загружается SBERT). Готовность:

```bash
curl http://127.0.0.1:8001/health
# {"status":"ok","pipeline_ready":true}
```

### 2. Go-gateway (порт 8080)

```bash
cd demo/gateway
go run .
```

Проверка:

```bash
curl http://127.0.0.1:8080/health
# {"gateway_status":"ok","ml_service_url":"http://127.0.0.1:8001","ml_status":"200 OK"}
```

### 3. Фронт

Открыть в браузере `http://127.0.0.1:8080` — gateway отдаёт статику из `demo/frontend/`.

Альтернатива (если не хочется поднимать gateway): открыть `demo/frontend/index.html`
напрямую как файл — JS сходит на `http://localhost:8080/api/enrich` (gateway всё
равно нужен).

## API

### `POST /api/enrich`

Тело запроса:

```json
{
  "category": "pasta",                       // pasta | chocolate | cheeses
  "product_name": "Spaghetti #5 Barilla",    // обязательное
  "brands": "Barilla",
  "ingredients_text": "Durum wheat semolina, water",
  "quantity": "500g"
}
```

Тело ответа:

```json
{
  "category": "pasta",
  "internal_category": "pasta_stratified",
  "n_attrs_total": 8,
  "n_covered": 7,
  "n_llm_fallback": 1,
  "predictions": {
    "grain_type":     { "value": "wheat",    "layer": "ml",    "confidence": 0.91 },
    "pasta_shape":    { "value": "spaghetti","layer": "regex", "confidence": 1.00 },
    "is_whole_grain": { "value": null,       "layer": "llm_fallback", "confidence": 0.0 }
  }
}
```

### `GET /api/categories`

Список доступных категорий и атрибутов.

### `GET /health`

Состояние gateway и ML-сервиса.

## Переменные окружения gateway

- `GATEWAY_ADDR` — адрес для прослушивания (по умолчанию `:8080`).
- `ML_SERVICE_URL` — URL ML-сервиса (по умолчанию `http://127.0.0.1:8001`).
- `FRONTEND_DIR` — директория со статическим фронтом (по умолчанию `../frontend`).
