# Система автоматизированного обогащения товарных данных для интернет-магазина

Выпускная квалификационная работа магистра, МАИ, 2026.

Гибридный каскадный конвейер автоматического заполнения атрибутов товаров: регулярные выражения → классификатор XGBoost на мультиязычных векторных представлениях → байесовская сеть → запасной слой LLM. Обучение и оценка — на открытых данных Open Food Facts.

## Запуск

```bash
brew install libomp                    # macOS: требуется для XGBoost
python3.14 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # sentence-transformers скачает ~2 ГБ весов
```

Для запасного слоя на LLM:

```bash
echo "OPENROUTER_API_KEY=sk-..." > .env
```

Воспроизведение всех ключевых чисел:

```bash
bash reproduce.sh --skip-llm           # без LLM, ~10 мин
bash reproduce.sh                      # полный прогон с LLM, ~30 мин
```

## Структура

- `report/` — текст ВКР (LaTeX, шаблон [iktovr/diploma-latex-template](https://github.com/iktovr/diploma-latex-template)).
  Сборка PDF: `cd report && make`.
- `slides/` — презентация к защите (LaTeX Beamer). Сборка: `cd slides && make`.
- `images/` — общая media-library: рисунки, графики, исходники (.dot, .pptx-экспорты).
  Используется и `report/`, и `slides/` через `\graphicspath{{../images/}}`.
- `notebooks/00_thesis_main.ipynb` — методология, эксперименты, рисунки.
- `src/` — исходный код конвейера (`pipeline/`, `eval/`, `diagnostics/`).
- `demo/` — рабочее демо: Go-шлюз + Python ML-сервис + фронтенд.
- `tests/` — модульные и интеграционные тесты.

## Сборка PDF ВКР

```bash
# Один раз: BasicTeX и нужные пакеты (потребуется sudo).
brew install --cask basictex
sudo bash report/scripts/install_packages.sh

# Каждый раз:
cd report && make
open main.pdf
```
