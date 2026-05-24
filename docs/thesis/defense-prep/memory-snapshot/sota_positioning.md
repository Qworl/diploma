---
name: SOTA positioning for thesis review/chapter 1
description: Какие SOTA работы по product attribute extraction и cost-aware LLM routing релевантны ВКР, у кого есть router, как позиционировать новизну каскада.
type: reference
originSessionId: 5e0d3d25-e69a-4195-8d68-319971cd8fcd
---
Для главы 1 (обзор) и Q&A на защите.

## Релевантные работы

| Работа | Router? | Что делает | Роль для ВКР |
|---|---|---|---|
| TXtract (Karamanolakis, ACL 2020) | Нет | Multi-task encoder с conditioning на таксономии категорий | Альтернатива Layer 2; нужно объяснить, почему выбран свой подход |
| MAVE (Yang, WSDM 2022) | Нет | Multi-source attribute extraction: BERT + cross-attention | Альтернатива Layer 2 |
| OpenTag (Zheng, KDD 2018) | Нет | BiLSTM-CRF + attention, sequence tagging | Альтернатива Layer 2 |
| **FrugalGPT** (Chen & Zaharia, ACL 2024) | **Да, центрально** | LLM cascade cheap→expensive, scorer на token probs | **Прямой аналог §6.14**; основной референс |
| AutoMix (Madaan, 2024) | Да | Self-verification routing LLM cascade | Цитировать |
| Hybrid LLM (Ding, ICLR 2024) | Да | Edge-cloud routing, BERT-классификатор сложности | Цитировать |
| RouterBench (Hu, NeurIPS 2024) | Benchmark | «No router beats best-single-model on all tasks» | **Прямая цитата в защиту negative result** |

## Позиционирование новизны

Каскад как идея — мейнстрим (FrugalGPT). Но FrugalGPT — single-paradigm каскад LLM↔LLM.

**Новизна ВКР:** гетерогенный каскад **rule (regex) → discriminative ML (XGBoost) → generative LLM**, с явным правилом per-attribute на маршрутизацию. Никакая из 4 ключевых работ не комбинирует все три парадигмы для product attribute enrichment.

## Готовая формулировка для §1.5

> Известные cost-aware каскады для языковых задач (FrugalGPT, AutoMix, Hybrid LLM) комбинируют однотипные генеративные модели разной мощности. В работе исследован гетерогенный каскад rule → discriminative ML → generative LLM с явным правилом маршрутизации на уровне (категория, атрибут). Конфигурация обоснована ablation-исследованиями на 6 разнородных доменах открытых каталогов с brand-disjoint оценкой и предзарегистрированной проверкой гипотезы о ценности обучаемого маршрутизатора.

## Согласование негативного результата H1 с RouterBench

Hu et al. (RouterBench) показали: универсальный router не доминирует на всех задачах. H1 ОТКЛОНЕНА на product attribute extraction — согласуется. Per-product сигнал в этой задаче слабее, чем в open-domain QA (где FrugalGPT работает). Это **уточняет** FrugalGPT, не противоречит.
