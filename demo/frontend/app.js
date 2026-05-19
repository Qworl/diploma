'use strict';

// Если фронт открыт через Go-gateway (http://localhost:8080), запросы идут к нему.
// Если открыт через file:// или другой статический сервер — указываем абсолютный URL.
const API_BASE = (window.location.protocol === 'file:' || window.location.port !== '8080')
  ? 'http://localhost:8080'
  : '';

const PRESETS = {
  pasta: {
    category: 'pasta',
    product_name: 'Spaghetti #5 Barilla',
    brands: 'Barilla',
    ingredients_text: 'Durum wheat semolina, water',
    quantity: '500g',
  },
  chocolate: {
    category: 'chocolate',
    product_name: 'Lindt Excellence Dark 70%',
    brands: 'Lindt',
    ingredients_text: 'Cocoa mass, sugar, cocoa butter, vanilla',
    quantity: '100g',
  },
  cheeses: {
    category: 'cheeses',
    product_name: 'Roquefort AOP',
    brands: 'Société',
    ingredients_text: 'Pasteurised sheep milk, salt, rennet, Penicillium roqueforti',
    quantity: '150g',
  },
  'validation-demo': {
    category: 'chocolate',
    product_name: 'Lindt Excellence White',
    brands: 'Lindt',
    ingredients_text: 'Sugar, cocoa butter, milk powder',
    quantity: '100g',
    expected: { cocoa_percentage: '70-85' },
  },
  'gluten-free-wheat': {
    category: 'pasta',
    product_name: 'Barilla Spaghetti #5',
    brands: 'Barilla',
    ingredients_text: 'Durum wheat semolina, water',
    quantity: '500g',
    expected: { is_gluten_free: true },
  },
  'garbage-name': {
    // Auto-режим: ловится Layer −1 (input-валидатор) до роутера.
    mode: 'auto',
    product_name: '??? @@@ 12345',
    brands: '',
    ingredients_text: '',
    quantity: '',
  },
  'empty-name': {
    mode: 'auto',
    product_name: '',
    brands: '',
    ingredients_text: '',
    quantity: '',
  },
};

const ATTR_LABELS = {
  // pasta
  grain_type: 'Тип зерна',
  pasta_shape: 'Форма пасты',
  is_whole_grain: 'Цельнозерновой',
  is_organic: 'Органик',
  is_gluten_free: 'Без глютена',
  is_vegan: 'Веганский',
  // common
  nutri_score_grade: 'Nutri-Score',
  protein_class: 'Класс белка',
  // chocolate
  chocolate_type: 'Тип шоколада',
  cocoa_percentage: 'Процент какао',
  contains_nuts: 'Содержит орехи',
  palm_oil_status: 'Пальмовое масло',
  // cheeses
  milk_source: 'Источник молока',
  texture: 'Текстура',
  country_of_origin: 'Страна происхождения',
  fat_class: 'Класс жирности',
  is_pdo: 'PDO/AOP',
  is_ultra_processed: 'Ультра-обработка',
};

const LAYER_LABELS = {
  regex: 'regex',
  ml: 'ML',
  bayes: 'байес',
  off_tags: 'OFF',
  llm_fallback: 'LLM-fallback',
  rejected_by_validator: 'отклонён валидатором',
};

const CATEGORY_ATTRS = {
  pasta: ['grain_type', 'pasta_shape', 'is_whole_grain', 'is_organic',
          'is_gluten_free', 'is_vegan', 'nutri_score_grade', 'protein_class'],
  chocolate: ['chocolate_type', 'cocoa_percentage', 'contains_nuts',
              'palm_oil_status', 'is_organic', 'nutri_score_grade', 'protein_class'],
  cheeses: ['milk_source', 'texture', 'country_of_origin', 'fat_class',
            'is_pdo', 'is_organic', 'is_ultra_processed',
            'nutri_score_grade', 'protein_class'],
};

// Per-category metadata: { [attr]: { kind: 'enum'|'bool'|'numeric_bins', states: [...] } }
// Populated from /api/categories on page load; null entries treated as free-form.
const ATTR_INFO = { pasta: {}, chocolate: {}, cheeses: {} };

async function loadCategoriesMetadata() {
  try {
    const resp = await fetch(`${API_BASE}/api/categories`);
    if (!resp.ok) return;
    const cats = await resp.json();
    cats.forEach(c => {
      const map = {};
      (c.attr_info || []).forEach(a => {
        if (a.states) map[a.name] = { kind: a.kind, states: a.states };
      });
      ATTR_INFO[c.category] = map;
    });
    // Re-render expected rows so dropdowns appear once metadata is available.
    renderExpectedRows(
      document.getElementById('category').value,
      loadExpectedFromStorage()
    );
  } catch (e) {
    console.warn('failed to load /api/categories:', e);
  }
}

const expectedRowsEl = document.getElementById('expected-rows');
const expectedAddBtn = document.getElementById('expected-add');

function buildValueControl(currentCategory, attr, value) {
  // Returns an <input> or <select> with class 'expected-value'.
  const info = (ATTR_INFO[currentCategory] || {})[attr];
  const kind = info ? info.kind : null;

  if (kind === 'enum' || kind === 'bool') {
    const sel = document.createElement('select');
    sel.className = 'expected-value';
    sel.dataset.kind = kind;
    const blank = document.createElement('option');
    blank.value = '';
    blank.textContent = '— выбрать —';
    sel.appendChild(blank);
    info.states.forEach(s => {
      const opt = document.createElement('option');
      opt.value = s;
      opt.textContent = s;
      if (String(value) === s) opt.selected = true;
      sel.appendChild(opt);
    });
    return sel;
  }

  const inp = document.createElement('input');
  inp.type = 'text';
  inp.className = 'expected-value';
  inp.value = value;
  if (kind === 'numeric_bins' && info.states.length) {
    inp.placeholder = `число (бакеты: ${info.states.join(', ')})`;
    inp.dataset.kind = 'numeric_bins';
  } else {
    inp.placeholder = 'значение';
    inp.dataset.kind = 'free';
  }
  return inp;
}

function renderExpectedRows(currentCategory, values) {
  expectedRowsEl.innerHTML = '';
  const attrs = CATEGORY_ATTRS[currentCategory] || [];
  Object.entries(values || {}).forEach(([attr, value]) => {
    const row = document.createElement('div');
    row.className = 'expected-row';
    const select = document.createElement('select');
    select.className = 'expected-attr';
    attrs.forEach(a => {
      const opt = document.createElement('option');
      opt.value = a;
      opt.textContent = ATTR_LABELS[a] || a;
      if (a === attr) opt.selected = true;
      select.appendChild(opt);
    });
    let valueControl = buildValueControl(currentCategory, attr, value);
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.textContent = '×';
    remove.addEventListener('click', () => {
      row.remove();
      saveExpectedToStorage();
    });
    // Swap the value control when the attribute changes.
    select.addEventListener('change', () => {
      const newControl = buildValueControl(currentCategory, select.value, '');
      newControl.addEventListener('input', saveExpectedToStorage);
      newControl.addEventListener('change', saveExpectedToStorage);
      row.replaceChild(newControl, valueControl);
      valueControl = newControl;
      saveExpectedToStorage();
    });
    valueControl.addEventListener('input', saveExpectedToStorage);
    valueControl.addEventListener('change', saveExpectedToStorage);
    row.appendChild(select);
    row.appendChild(valueControl);
    row.appendChild(remove);
    expectedRowsEl.appendChild(row);
  });
}

function readExpected() {
  const out = {};
  expectedRowsEl.querySelectorAll('.expected-row').forEach(row => {
    const attr = row.querySelector('.expected-attr').value;
    const valueEl = row.querySelector('.expected-value');
    if (!attr || !valueEl) return;
    const raw = String(valueEl.value || '').trim();
    if (!raw) return;
    const kind = valueEl.dataset.kind || 'free';
    if (kind === 'bool') {
      out[attr] = (raw === 'True');
      return;
    }
    if (kind === 'enum') {
      out[attr] = raw;
      return;
    }
    // free / numeric_bins: numeric heuristic.
    const num = Number(raw);
    out[attr] = (!isNaN(num) && raw === String(num)) ? num : raw;
  });
  return out;
}

function saveExpectedToStorage() {
  localStorage.setItem('expected_values', JSON.stringify(readExpected()));
}

function loadExpectedFromStorage() {
  try {
    return JSON.parse(localStorage.getItem('expected_values') || '{}');
  } catch (e) {
    return {};
  }
}

expectedAddBtn.addEventListener('click', () => {
  const cat = document.getElementById('category').value;
  const attrs = CATEGORY_ATTRS[cat] || [];
  if (attrs.length === 0) return;
  const current = readExpected();
  // Pick first attribute not already in the list.
  const unused = attrs.find(a => !(a in current));
  if (!unused) return;
  current[unused] = '';
  renderExpectedRows(cat, current);
});

document.getElementById('category').addEventListener('change', () => {
  const cat = document.getElementById('category').value;
  renderExpectedRows(cat, loadExpectedFromStorage());
});

// Initial render.
renderExpectedRows(document.getElementById('category').value, loadExpectedFromStorage());

// Category mode radio toggle.
const categorySelect = document.getElementById("category");
const modeRadios = document.querySelectorAll('input[name="category_mode"]');
modeRadios.forEach((r) => {
  r.addEventListener("change", () => {
    const isAuto = document.querySelector('input[name="category_mode"]:checked').value === "auto";
    categorySelect.disabled = isAuto;
  });
});
categorySelect.disabled = true;

const form = document.getElementById('enrich-form');
const submitBtn = document.getElementById('submit-btn');
const resultCard = document.getElementById('result-card');
const errorCard = document.getElementById('error-card');
const errorText = document.getElementById('error-text');
const predictionsBody = document.getElementById('predictions-body');
const rawJson = document.getElementById('raw-json');

document.querySelectorAll('.presets button').forEach(btn => {
  btn.addEventListener('click', () => applyPreset(btn.dataset.preset));
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  await runEnrich();
});

function applyPreset(name) {
  const p = PRESETS[name];
  if (!p) return;
  const wantAuto = p.mode === 'auto' || !p.category;
  const targetMode = wantAuto ? 'auto' : 'manual';
  const radio = document.querySelector(`input[name="category_mode"][value="${targetMode}"]`);
  if (radio) {
    radio.checked = true;
    categorySelect.disabled = wantAuto;
  }
  if (p.category) {
    document.getElementById('category').value = p.category;
  }
  document.getElementById('product_name').value = p.product_name;
  document.getElementById('brands').value = p.brands;
  document.getElementById('ingredients_text').value = p.ingredients_text;
  document.getElementById('quantity').value = p.quantity;
  const expected = p.expected || {};
  localStorage.setItem('expected_values', JSON.stringify(expected));
  // Без явной категории expected-блок отрисовать не от чего — оставляем пустым.
  renderExpectedRows(p.category || document.getElementById('category').value, expected);
}

async function runEnrich() {
  hideCards();
  submitBtn.disabled = true;
  submitBtn.textContent = 'Считаю ...';
  const t0 = performance.now();

  const isAuto = document.querySelector('input[name="category_mode"]:checked').value === "auto";
  const data = {
    product_name: document.getElementById('product_name').value,
    brands: document.getElementById('brands').value,
    ingredients_text: document.getElementById('ingredients_text').value,
    quantity: document.getElementById('quantity').value,
    validate: 'warn',
    expected: readExpected(),
    fallback_on_ood: document.getElementById('fallback-on-ood').checked,
  };
  if (!isAuto) {
    data.category = document.getElementById('category').value;
  }

  try {
    const resp = await fetch(`${API_BASE}/api/enrich`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    const elapsed = Math.round(performance.now() - t0);
    const body = await resp.json();
    if (!resp.ok) {
      showError(body.error || `HTTP ${resp.status}`);
      return;
    }
    renderResult(body, elapsed);
  } catch (err) {
    showError(err.message || String(err));
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = 'Обогатить';
  }
}

function hideCards() {
  resultCard.hidden = true;
  errorCard.hidden = true;
}

function showError(msg) {
  errorText.textContent = msg;
  errorCard.hidden = false;
}

function renderCategoryInference(data) {
  const infBlock = document.getElementById("meta-category-inference");
  const oodCard = document.getElementById("ood-card");
  const unsupCard = document.getElementById("unsupported-card");
  const invalidCard = document.getElementById("invalid-input-card");
  oodCard.hidden = true;
  unsupCard.hidden = true;
  if (invalidCard) invalidCard.hidden = true;
  infBlock.hidden = true;

  if (data.is_invalid_input) {
    if (invalidCard) {
      invalidCard.hidden = false;
      document.getElementById("invalid-input-reason").textContent =
        (data.invalid_input && data.invalid_input.reason) || "?";
      document.getElementById("invalid-input-message").textContent =
        (data.invalid_input && data.invalid_input.message) || "";
    }
    return;
  }
  if (data.is_ood) {
    oodCard.hidden = false;
    const top3Line = document.getElementById("ood-top3-line");
    const semLine = document.getElementById("ood-semantic-line");
    const reasonEl = document.getElementById("ood-reason");
    if (data.semantic_ood) {
      // Отбили на Layer 0.5 — softmax-роутера вообще не было.
      reasonEl.textContent =
        "Семантический OOD: эмбеддинг товара слишком далёк от всех известных центроидов классов.";
      if (top3Line) top3Line.hidden = true;
      if (semLine) {
        semLine.hidden = false;
        document.getElementById("ood-semantic-dist").textContent =
          data.semantic_ood.distance.toFixed(2);
        document.getElementById("ood-semantic-thr").textContent =
          data.semantic_ood.threshold.toFixed(2);
        document.getElementById("ood-semantic-nearest").textContent =
          data.semantic_ood.nearest_class;
      }
    } else {
      reasonEl.textContent = "Товар не относится к поддерживаемым категориям.";
      if (top3Line) top3Line.hidden = false;
      if (semLine) semLine.hidden = true;
      document.getElementById("ood-top3").textContent =
        (data.category_inference?.alternatives || [])
          .map((alt) => `${alt[0]} (${alt[1].toFixed(2)})`).join(", ");
    }
    return;
  }
  if (data.is_known_but_unsupported) {
    unsupCard.hidden = false;
    document.getElementById("unsupported-cat").textContent =
      data.category_inference?.predicted || "?";
    return;
  }
  if (data.category_inference) {
    infBlock.hidden = false;
    document.getElementById("meta-cat-pred").textContent =
      data.category_inference.predicted;
    document.getElementById("meta-cat-conf").textContent =
      data.category_inference.confidence.toFixed(2);
    document.getElementById("meta-cat-ood").textContent =
      String(data.category_inference.is_ood);
    const ul = document.getElementById("meta-cat-alts");
    ul.innerHTML = "";
    (data.category_inference.alternatives || []).forEach((alt) => {
      const li = document.createElement("li");
      li.textContent = `${alt[0]}: ${alt[1].toFixed(3)}`;
      ul.appendChild(li);
    });
  }
}

function renderResult(body, elapsedMs) {
  document.getElementById('meta-category').textContent = body.category ?? '—';
  document.getElementById('meta-covered').textContent = body.n_covered;
  document.getElementById('meta-total').textContent = body.n_attrs_total;
  document.getElementById('meta-fallback').textContent = body.n_llm_fallback;
  document.getElementById('meta-duration').textContent = `${elapsedMs} мс`;

  renderCategoryInference(body);

  if (body.is_invalid_input) {
    predictionsBody.innerHTML = '';
    const badge = document.getElementById('summary-badge');
    if (badge) badge.hidden = true;
    const brandEl = document.getElementById('brand-status');
    if (brandEl) brandEl.hidden = true;
    const detailsEl = document.querySelector('.validation-details');
    if (detailsEl) detailsEl.open = false;
    rawJson.textContent = JSON.stringify(body, null, 2);
    resultCard.hidden = false;
    return;
  }

  // Detect frontend-side issues that the backend's flag-rate counters don't see:
  // expected values provided but not projectable to the network's domain.
  const nInvalidExpected = Object.values(body.expected || {}).filter(e =>
    e && e.value !== null && e.value !== undefined && e.value !== '' && e.bucketized_to == null
  ).length;

  // For each flagged prediction, decide whether the conflict is more plausibly
  // attributable to the user (their expected value appears as a top-negative
  // PMI contributor) or to the model (predictions internally inconsistent).
  // Heuristic: if any top-negative PMI contributor is one of the user's
  // expected attrs → "user", otherwise "model".
  const expectedAttrSet = new Set(
    Object.entries(body.expected || {})
      .filter(([, e]) => e && e.value !== null && e.value !== undefined && e.value !== '')
      .map(([attr]) => attr)
  );
  const attribution = {}; // { [attr]: 'model' | 'user' }
  let nAttribModel = 0;
  let nAttribUser = 0;
  for (const [attr, p] of Object.entries(body.predictions)) {
    if (!(p.validation && p.validation.flagged)) continue;
    const conflictsWithUserAttr = (p.validation.contributors || [])
      .find(c => (c.pmi ?? 0) < 0 && expectedAttrSet.has(c.attr));
    if (conflictsWithUserAttr) {
      attribution[attr] = { source: 'user', via: conflictsWithUserAttr };
      nAttribUser += 1;
    } else {
      attribution[attr] = { source: 'model' };
      nAttribModel += 1;
    }
  }
  // Expected-side flags are by definition "user" (whether they tipped a
  // border-case prediction or not, the explicit user value is below threshold).
  for (const [, e] of Object.entries(body.expected || {})) {
    if (e && e.validation && e.validation.flagged) nAttribUser += 1;
  }

  renderBrandStatus(body.validation_summary);
  renderSummaryBadge({
    nFlaggedModel: nAttribModel,
    nFlaggedUser: nAttribUser,
    nInvalid: nInvalidExpected,
  });

  predictionsBody.innerHTML = '';
  for (const attr of Object.keys(body.predictions)) {
    const pred = body.predictions[attr];
    const exp = body.expected[attr] || null;
    const tr = renderRow(attr, pred, exp, attribution[attr]);
    predictionsBody.appendChild(tr);
  }

  renderValidationDetails(body);
  // Auto-expand the details section when there is anything to explain.
  const detailsEl = document.querySelector('.validation-details');
  if (detailsEl) {
    detailsEl.open = (nAttribModel + nAttribUser + nInvalidExpected) > 0;
  }

  rawJson.textContent = JSON.stringify(body, null, 2);
  resultCard.hidden = false;
}

function renderBrandStatus(summary) {
  const el = document.getElementById('brand-status');
  if (!el || !summary) return;
  const dot = el.querySelector('.brand-status-dot');
  const txt = el.querySelector('.brand-status-text');
  if (summary.brand_status === 'known') {
    dot.style.background = 'var(--color-ok, #2ecc71)';
    txt.textContent = 'brand: known in training distribution';
  } else if (summary.brand_status === 'ood') {
    dot.style.background = 'var(--color-warn, #f39c12)';
    txt.textContent = 'brand: out of training distribution';
  } else {
    dot.style.background = '#999';
    txt.textContent = 'brand: n/a (no brand node in validator network)';
  }
  el.hidden = false;
}

function renderSummaryBadge(stats) {
  const el = document.getElementById('summary-badge');
  if (!el || !stats) return;
  const { nFlaggedModel = 0, nFlaggedUser = 0, nInvalid = 0 } = stats;
  const total = nFlaggedModel + nFlaggedUser;
  if (total === 0 && nInvalid === 0) {
    el.className = 'summary-badge summary-ok';
    el.textContent = 'Все атрибуты согласованы';
  } else {
    el.className = 'summary-badge summary-flag';
    const parts = [];
    if (total > 0) {
      parts.push(`Найдено противоречий: ${total} `
        + `(модель: ${nFlaggedModel}, пользователь: ${nFlaggedUser})`);
    }
    if (nInvalid > 0) {
      parts.push(`Невалидных значений от пользователя: ${nInvalid}`);
    }
    el.textContent = parts.join('; ');
  }
  el.hidden = false;
}

function renderRow(attr, pred, exp, attribution) {
  const tr = document.createElement('tr');
  const predFlagged = pred.validation && pred.validation.flagged;
  const expFlagged = exp && exp.validation && exp.validation.flagged;
  if (predFlagged || expFlagged) tr.classList.add('row-flagged');

  // Column 1: attribute label.
  const td1 = document.createElement('td');
  td1.textContent = ATTR_LABELS[attr] || attr;
  td1.title = attr;
  tr.appendChild(td1);

  // Column 2: ML/regex prediction.
  const td2 = document.createElement('td');
  td2.textContent = formatValue(pred.value);
  if (pred.confidence > 0) {
    const conf = document.createElement('div');
    conf.className = 'muted small';
    conf.textContent = `confidence ${(pred.confidence * 100).toFixed(0)}%`;
    td2.appendChild(conf);
  }
  tr.appendChild(td2);

  // Column 3: expected.
  const td3 = document.createElement('td');
  const expValueProvided = exp && exp.value !== null && exp.value !== undefined && exp.value !== '';
  const expInvalid = expValueProvided && exp.bucketized_to == null;
  if (exp) {
    td3.textContent = formatValue(exp.value);
    // Show the bucketized projection only when it differs from the raw input
    // (i.e. there was a non-trivial coercion: "True"<-"Да", 70<-"70-85", ...).
    const shouldShowBucket = exp.bucketized_to && String(exp.bucketized_to) !== String(exp.value);
    if (shouldShowBucket) {
      const bucket = document.createElement('div');
      bucket.className = 'muted small';
      bucket.textContent = `→ ${exp.bucketized_to}`;
      td3.appendChild(bucket);
    } else if (expInvalid) {
      const warn = document.createElement('div');
      warn.className = 'val-flag small';
      warn.textContent = 'значение не приводится к домену атрибута';
      td3.appendChild(warn);
      tr.classList.add('row-flagged');
    }
    if (exp.agrees_with_predicted === false && !expInvalid) {
      td3.classList.add('disagrees');
    }
  } else {
    td3.textContent = '—';
    td3.classList.add('muted');
  }
  tr.appendChild(td3);

  // Column 4: layer.
  const td4 = document.createElement('td');
  const tag = document.createElement('span');
  tag.className = `layer-tag layer-${pred.layer}`;
  tag.textContent = LAYER_LABELS[pred.layer] || pred.layer;
  td4.appendChild(tag);
  tr.appendChild(td4);

  // Column 5: validation badge.
  const td5 = document.createElement('td');
  td5.appendChild(renderValidationCell(
    pred.validation,
    exp ? exp.validation : null,
    expInvalid,
    attribution,
  ));
  tr.appendChild(td5);

  return tr;
}

function _renderLine(label, v) {
  // One validation line: "pred: ok P=0.587" or "exp: flagged P=0.030".
  const cls = v.flagged ? 'val-flag' : 'val-ok';
  const verdict = v.flagged ? 'flagged' : 'ok';
  return `<div class="val-line"><span class="val-label muted small">${label}</span> `
       + `<span class="${cls}">${verdict}</span> P=${v.p.toFixed(3)}</div>`;
}

function renderValidationCell(predV, expV, expInvalid, attribution) {
  const wrap = document.createElement('div');
  wrap.className = 'val-cell';
  if (expInvalid) {
    wrap.innerHTML = `<span class="val-flag">невалидный ввод</span>`;
    return wrap;
  }
  if (predV == null && expV == null) {
    wrap.textContent = '—';
    wrap.classList.add('muted');
    return wrap;
  }

  let html = '';
  // Always render both lines when both sides exist; single line otherwise.
  if (predV != null && expV != null) {
    html += _renderLine('предсказание', predV);
    html += _renderLine('ввод', expV);
  } else if (predV != null) {
    html += _renderLine('предсказание', predV);
  } else {
    html += _renderLine('ввод', expV);
  }

  // Conflict hint: shown once below the lines.
  const flagged = (predV && predV.flagged) ? predV : (expV && expV.flagged) ? expV : null;
  if (flagged) {
    if (attribution && attribution.source === 'user' && attribution.via) {
      const v = attribution.via;
      html += `<div class="muted small">конфликт с вводом: ${v.attr}=${v.value}</div>`;
    } else {
      const top = (flagged.contributors || []).filter(c => (c.pmi ?? 0) < 0)[0];
      if (top) {
        const times = Math.round(Math.exp(-top.pmi));
        html += `<div class="muted small">конфликт: ${top.attr}=${top.value} (×${times})</div>`;
      }
    }
  }
  wrap.innerHTML = html;
  return wrap;
}

function renderValidationDetails(body) {
  const el = document.getElementById('validation-details-body');
  el.innerHTML = '';
  const flagged = [];
  for (const [attr, p] of Object.entries(body.predictions)) {
    if (p.validation && p.validation.flagged) {
      flagged.push({ source: 'predicted', attr, value: p.value, v: p.validation });
    }
  }
  for (const [attr, e] of Object.entries(body.expected)) {
    if (e.validation && e.validation.flagged) {
      flagged.push({ source: 'expected', attr, value: e.value, v: e.validation });
    }
  }
  if (flagged.length === 0) {
    el.innerHTML = '<p class="muted">Нет флагов.</p>';
    return;
  }
  for (const f of flagged) {
    const card = document.createElement('div');
    card.className = 'val-detail-card';
    let html = `<h4>${f.attr} = ${f.value} <span class="muted small">(${f.source})</span></h4>`;
    html += `<p>P = ${f.v.p.toFixed(4)}, marginal P = ${f.v.marginal_p.toFixed(4)}, `;
    html += `threshold = ${f.v.threshold.toFixed(4)}</p>`;
    html += '<table class="pmi-table"><thead><tr><th>evidence</th><th>PMI</th></tr></thead><tbody>';
    for (const c of (f.v.contributors || [])) {
      html += `<tr><td>${c.attr}=${c.value}</td><td>${c.pmi.toFixed(3)}</td></tr>`;
    }
    html += '</tbody></table>';
    html += `<button class="shapley-btn" data-attr="${f.attr}" data-value="${f.value}">Посчитать вклад свидетельств (Shapley)</button>`;
    html += `<div class="shapley-result"></div>`;
    card.innerHTML = html;
    el.appendChild(card);
  }
  // Wire shapley buttons.
  el.querySelectorAll('.shapley-btn').forEach(btn => {
    btn.addEventListener('click', () => requestShapley(btn, body));
  });
}

async function requestShapley(btn, body) {
  btn.disabled = true;
  const originalLabel = btn.textContent;
  btn.textContent = 'Считаю ...';
  const target = btn.dataset.attr;
  const value = btn.dataset.value;
  const evidence = {};
  for (const [a, p] of Object.entries(body.predictions)) {
    if (a !== target && p.value !== null) evidence[a] = p.value;
  }
  try {
    const resp = await fetch(`${API_BASE}/api/explain`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        category: body.category,
        brands: document.getElementById('brands').value,
        attribute: target,
        value: isNaN(Number(value)) ? value : Number(value),
        evidence,
        shapley_mode: 'sampled',
        shapley_samples: 100,
      }),
    });
    const data = await resp.json();
    const out = btn.parentElement.querySelector('.shapley-result');
    out.innerHTML = renderShapleyTable(data);
    // Wire the toggle that reveals near-zero contributors.
    const toggle = out.querySelector('.shapley-toggle');
    if (toggle) {
      toggle.addEventListener('click', () => {
        out.querySelectorAll('.shapley-row-zero').forEach(r => r.classList.toggle('hidden'));
        toggle.textContent = toggle.textContent.startsWith('показать')
          ? 'скрыть свидетельства со слабым вкладом'
          : 'показать все свидетельства';
      });
    }
  } finally {
    btn.disabled = false;
    btn.textContent = originalLabel;
  }
}

function renderShapleyTable(data) {
  const rows = (data.attribution || []).slice();
  // Sort by |Shapley| descending — strongest drivers first.
  rows.sort((a, b) => Math.abs(b.shapley) - Math.abs(a.shapley));
  // Threshold for "weak contribution" — hidden by default behind a toggle.
  const EPSILON = 0.01;
  // Max |Shapley| for bar scaling.
  const maxAbs = rows.reduce((m, r) => Math.max(m, Math.abs(r.shapley)), 0) || 1;

  let html = '<table class="pmi-table shapley-table"><thead><tr>'
    + '<th>свидетельство</th><th>PMI</th><th class="shapley-col">Shapley</th>'
    + '</tr></thead><tbody>';
  let nWeak = 0;
  for (const row of rows) {
    const isWeak = Math.abs(row.shapley) < EPSILON;
    if (isWeak) nWeak += 1;
    const cls = isWeak ? ' class="shapley-row-zero hidden"' : '';
    const pmi = row.pmi ?? 0;
    const shap = row.shapley;
    const barPct = Math.min(100, (Math.abs(shap) / maxAbs) * 100);
    const barColor = shap < 0 ? '#e74c3c' : '#2ecc71';
    const align = shap < 0 ? 'flex-end' : 'flex-start';
    const bar = `<span class="shapley-bar-wrap" style="justify-content:${align};">`
      + `<span class="shapley-bar" style="width:${barPct.toFixed(1)}%;background:${barColor};"></span>`
      + `<span class="shapley-num">${shap.toFixed(3)}</span></span>`;
    html += `<tr${cls}>`
      + `<td>${row.evidence}</td>`
      + `<td>${pmi.toFixed(3)}</td>`
      + `<td class="shapley-col">${bar}</td>`
      + '</tr>';
  }
  html += '</tbody></table>';

  if (nWeak > 0) {
    html += `<button type="button" class="shapley-toggle small muted">`
      + `показать все свидетельства (+${nWeak} со слабым вкладом)</button>`;
  }
  if (data.shapley_efficiency_check) {
    const c = data.shapley_efficiency_check;
    html += `<p class="muted small shapley-efficiency">`
      + `проверка эффективности (axiom): sum=${c.sum_shapley.toFixed(3)}, `
      + `log P-diff=${c.log_likelihood_diff.toFixed(3)}, `
      + `остаток=${c.residual.toFixed(3)} `
      + `(должен быть ≈ 0)</p>`;
  }
  return html;
}

function formatValue(v) {
  if (v === null || v === undefined) return '— (не закрыто, нужен LLM-fallback)';
  if (typeof v === 'boolean') return v ? 'Да' : 'Нет';
  return String(v);
}

function formatConfidence(c) {
  const wrapper = document.createElement('span');
  wrapper.textContent = `${(c * 100).toFixed(0)}%`;

  if (c > 0) {
    const bar = document.createElement('span');
    bar.className = 'confidence-bar';
    const fill = document.createElement('span');
    fill.style.width = `${Math.round(c * 100)}%`;
    bar.appendChild(fill);
    wrapper.appendChild(bar);
  }
  return wrapper;
}

// Авто-применение pasta preset при первом открытии
applyPreset('pasta');

// Подгружаем метаданные атрибутов (state_names из обученных Bayes-сетей),
// чтобы для enum/bool атрибутов в expected-блоке показывать <select>.
loadCategoriesMetadata();
