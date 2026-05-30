'use strict';

// Если фронт открыт через Go-gateway (http://localhost:8080), запросы идут к нему.
// Если открыт через file:// или другой статический сервер — указываем абсолютный URL.
const API_BASE = (window.location.protocol === 'file:' || window.location.port !== '8080')
  ? 'http://localhost:8080'
  : '';

// Порог: confidence ≥ AUTO_ACCEPT → принимается системой автоматически, иначе
// попадает в «требует проверки». Real per-attr thresholds живут в
// {category}_thresholds.pkl, но frontend их не видит.
const AUTO_ACCEPT = 0.90;

// Демо-пресеты: «входные» данные карточки от партнёра. До нажатия «Дозаполнить»
// показывается только эта часть; всё остальное (ML/Bayes/LLM-fallback) висит как
// пустое «требует проверки» — оператор сам решает, когда запустить каскад.
const PRESETS = {
  chocolate: {
    category: 'chocolate',
    product_name: 'Lindt Excellence Dark 70% Cocoa',
    brands: 'Lindt',
    ingredients_text: 'Cocoa mass, sugar, cocoa butter, soya lecithin, vanilla extract',
    quantity: '100 g',
    sku: 'LNDT-EXC-D70',
    image_url: 'images/chocolate.jpg',
  },
  pasta: {
    category: 'pasta',
    product_name: 'Spaghetti #5 Barilla',
    brands: 'Barilla',
    ingredients_text: 'Durum wheat semolina, water',
    quantity: '500 g',
    sku: 'BRL-SP5-500',
    image_url: 'images/pasta.jpg',
  },
  cheeses: {
    category: 'cheeses',
    product_name: 'Roquefort AOP',
    brands: 'Société',
    ingredients_text: 'Pasteurised sheep milk, salt, rennet, Penicillium roqueforti',
    quantity: '150 g',
    sku: 'STE-RQF-150',
    image_url: 'images/cheese.jpg',
  },
};

// Локализация имён атрибутов. Под ATTR_LABELS_RU[attr] лежит русское имя;
// если нет в карте — fallback на snake_case (показывается как есть).
const ATTR_LABELS_RU = {
  brand: 'Бренд',
  net_weight: 'Масса нетто',
  composition: 'Состав',
  category: 'Категория',
  // pasta
  grain_type: 'Тип зерна',
  pasta_shape: 'Форма пасты',
  is_filled: 'С начинкой',
  is_gluten_free: 'Без глютена',
  is_vegan: 'Веганский',
  cuisine_origin: 'Кухня',
  // chocolate
  chocolate_type: 'Тип шоколада',
  contains_nuts: 'Содержит орехи',
  chocolate_extra: 'Добавки',
  flavor_profile: 'Вкусовой профиль',
  // cheeses
  milk_source: 'Источник молока',
  texture: 'Текстура',
  country_of_origin: 'Страна происхождения',
  aging: 'Выдержка',
  is_pdo: 'PDO/AOP',
  is_ultra_processed: 'Ультра-обработанный',
  // common
  is_organic: 'Органик',
};

// Локализация значений атрибутов: { attr: { value: 'русский лейбл', ... } }.
// Boolean идут по общей ветке formatBool.
const VALUE_LABELS_RU = {
  milk_source: {
    cow: 'коровье', goat: 'козье', sheep: 'овечье', buffalo: 'буйволиное',
    mixed: 'смесь', other: 'другое',
  },
  texture: {
    hard: 'твёрдая', soft: 'мягкая', fresh: 'свежая', cream: 'сливочная',
    blue: 'с плесенью', processed: 'плавленый', other: 'другая',
  },
  country_of_origin: {
    france: 'Франция', italy: 'Италия', spain: 'Испания', germany: 'Германия',
    uk: 'Великобритания', us: 'США', switzerland: 'Швейцария',
    netherlands: 'Нидерланды', greece: 'Греция', denmark: 'Дания',
    cyprus: 'Кипр', india: 'Индия', mexico: 'Мексика', belgium: 'Бельгия',
    bulgaria: 'Болгария', ireland: 'Ирландия', norway: 'Норвегия',
    russia: 'Россия', other: 'другая',
  },
  aging: { fresh: 'свежий', young: 'молодой', aged: 'выдержанный' },
  chocolate_type: {
    dark: 'тёмный', milk: 'молочный', white: 'белый',
    filled: 'с начинкой', other: 'другой',
  },
  chocolate_extra: {
    plain: 'без добавок', with_nuts: 'с орехами', with_fruit: 'с фруктами',
    with_caramel: 'с карамелью', with_cookie: 'с печеньем',
    filled: 'с начинкой', with_alcohol: 'с алкоголем',
    with_coffee: 'с кофе', other: 'другое',
  },
  flavor_profile: {
    sweet_creamy: 'сливочно-сладкий', intense_bitter: 'горький',
    fruity: 'фруктовый', spiced: 'пряный',
    salty_caramel: 'солёная карамель', nutty: 'ореховый',
    floral: 'цветочный', other: 'другой',
  },
  grain_type: {
    wheat: 'пшеница', spelt: 'полба', rice: 'рис', corn: 'кукуруза',
    buckwheat: 'гречиха', oat: 'овёс', potato: 'картофель',
    legume: 'бобовые', mixed: 'смесь', other: 'другое',
  },
  pasta_shape: {
    spaghetti: 'спагетти', penne: 'пенне', fusilli: 'фузилли',
    macaroni: 'макароны', farfalle: 'фарфалле', tagliatelle: 'тальятелле',
    lasagna: 'лазанья', noodles: 'лапша', rigatoni: 'ригатони',
    vermicelli: 'вермишель', linguine: 'лингвини', shells: 'ракушки',
    gnocchi: 'ньокки', orzo: 'орзо', other: 'другое',
  },
  cuisine_origin: {
    italian: 'итальянская', asian: 'азиатская',
    german_alpine: 'немецко-альпийская', other_regional: 'другая региональная',
    other: 'другое',
  },
  category: { chocolate: 'Шоколад', pasta: 'Паста', cheeses: 'Сыры' },
};

// Per-category metadata из /api/categories: { [attr]: { kind, states } }.
const ATTR_INFO = { pasta: {}, chocolate: {}, cheeses: {} };
// Список ML-атрибутов категории (нужен для рендера пустой карточки до enrich).
const CATEGORY_ATTRS = { pasta: [], chocolate: [], cheeses: [] };

// Локальный state операторской сессии.
const state = {
  preset: 'chocolate',
  product: null,             // объект из PRESETS
  response: null,            // последний ответ от /api/enrich (null до первого «Дозаполнить»)
  confirmed: {},             // attr → value, что оператор зафиксировал
  rejected: new Set(),       // attr — оператор удалил предложение
  edited: {},                // attr → current input value (live)
  pendingAttrs: [],          // порядок pending-атрибутов
};

// ---------- API ----------

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
      CATEGORY_ATTRS[c.category] = c.attrs || [];
    });
  } catch (e) {
    console.warn('failed to load /api/categories:', e);
  }
}

async function callEnrich(preset, confirmed) {
  const payload = {
    category: preset.category,
    product_name: preset.product_name,
    brands: preset.brands,
    ingredients_text: preset.ingredients_text,
    quantity: preset.quantity,
    validate: 'warn',
    confirmed: confirmed || {},
  };
  const resp = await fetch(`${API_BASE}/api/enrich`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}));
    throw new Error(body.error || `HTTP ${resp.status}`);
  }
  return await resp.json();
}

// ---------- Helpers ----------

function attrLabel(attr) {
  return ATTR_LABELS_RU[attr] || attr;
}

function formatBool(v) {
  if (v === true || v === 'True' || v === 'true' || v === 'Да') return 'Да';
  if (v === false || v === 'False' || v === 'false' || v === 'Нет') return 'Нет';
  return null;
}

function formatValue(attr, v) {
  if (v === null || v === undefined || v === '') return '—';
  const b = formatBool(v);
  if (b !== null) return b;
  const map = VALUE_LABELS_RU[attr];
  if (map && map[v] !== undefined) return map[v];
  return String(v);
}

function isRowEdited(row) {
  // «Отредактировано» = у системы реально было предложение И оператор
  // ввёл другое значение. Если pred.value == null (каскад ещё не
  // запускался, или атрибут оставлен для LLM-fallback) — ввод оператора
  // просто заполняет пустоту, это не конфликт.
  if (row.kind !== 'pending') return false;
  const sys = row.pred.value;
  if (sys === null || sys === undefined || sys === '') return false;
  const edited = state.edited[row.attr];
  if (edited == null || edited === '') return false;
  return String(edited) !== String(sys);
}

function isAuto(pred) {
  if (pred.value === null || pred.value === undefined) return false;
  if (pred.layer === 'regex' || pred.layer === 'off_tags') return true;
  if (pred.layer === 'operator') return true;
  if (pred.layer === 'llm_fallback') return false;
  return pred.confidence >= AUTO_ACCEPT;
}

function pmiTimesPhrase(validation) {
  if (!validation || !validation.contributors) return null;
  const top = validation.contributors.filter(c => (c.pmi ?? 0) < 0)[0];
  if (!top) return null;
  const times = Math.max(2, Math.round(Math.exp(-top.pmi)));
  return { times, attr: top.attr, value: top.value };
}

function makeLayerPill(layer) {
  const span = document.createElement('span');
  const cls = layer === 'from_card' ? 'layer-from_card' : `layer-${layer}`;
  span.className = `layer-pill ${cls}`;
  span.textContent = layer === 'from_card' ? 'из карточки'
                   : layer === 'operator' ? 'оператор'
                   : layer;
  return span;
}

// ---------- Rendering ----------

function render() {
  const product = state.product;
  const r = state.response;  // null до первого «Дозаполнить»

  // hero
  document.getElementById('hero-category').textContent = product.category;
  const cConf = r?.category_inference?.confidence;
  document.getElementById('hero-category-conf').textContent =
    (cConf != null ? cConf : 1.00).toFixed(2);
  document.getElementById('hero-title').textContent = product.product_name;
  document.getElementById('hero-brand').textContent = product.brands;
  document.getElementById('hero-quantity').textContent = product.quantity;
  document.getElementById('hero-sku').textContent = product.sku || '—';

  // Сборка строк таблицы.
  const allRows = [];

  // 1. Поля «из карточки» — partner-side, всегда заполнены.
  allRows.push({ kind: 'card', attr: 'brand', value: product.brands });
  allRows.push({ kind: 'card', attr: 'net_weight', value: product.quantity });
  allRows.push({ kind: 'card', attr: 'composition', value: product.ingredients_text });

  // 2. Категория. До enrich — берётся из preset (manual mode); после — из ответа.
  allRows.push({
    kind: 'auto',
    attr: 'category',
    value: r ? r.category : product.category,
    layer: 'ml',
    confidence: r?.category_inference?.confidence ?? 1.0,
  });

  // 3. ML/Bayes/LLM атрибуты.
  const attrs = CATEGORY_ATTRS[product.category] || [];
  for (const attr of attrs) {
    const pred = r?.predictions?.[attr];
    if (state.rejected.has(attr)) {
      allRows.push({ kind: 'pending', attr, pred: { value: null, layer: 'llm_fallback', confidence: 0, validation: null } });
      continue;
    }
    if (!pred) {
      // До первого enrich — все атрибуты пустые pending.
      allRows.push({
        kind: 'pending',
        attr,
        pred: { value: null, layer: 'llm_fallback', confidence: 0, validation: null },
      });
      continue;
    }
    if (pred.value === null || pred.value === undefined) {
      allRows.push({ kind: 'pending', attr, pred });
    } else if (isAuto(pred)) {
      allRows.push({
        kind: 'auto', attr,
        value: pred.value, layer: pred.layer, confidence: pred.confidence,
      });
    } else {
      allRows.push({ kind: 'pending', attr, pred });
    }
  }

  // Подсчёты для пончика/пилюль.
  const cardCount = allRows.filter(r => r.kind === 'card').length;
  const autoCount = allRows.filter(r => r.kind === 'auto').length;
  const pendingRows = allRows.filter(r => r.kind === 'pending');
  const confirmedCount = cardCount + autoCount;
  const total = allRows.length;
  state.pendingAttrs = pendingRows.map(r => r.attr);

  document.getElementById('donut-fraction').textContent = `${confirmedCount}/${total}`;
  const pct = total > 0 ? (confirmedCount / total * 100) : 0;
  document.getElementById('hero-donut').style.setProperty('--pct', pct.toFixed(0));
  document.getElementById('stat-card').textContent = cardCount;
  document.getElementById('stat-auto').textContent = autoCount;
  document.getElementById('stat-pending').textContent = pendingRows.length;

  // pills
  const pillPending = document.getElementById('pill-pending');
  if (pendingRows.length > 0) {
    document.getElementById('pill-pending-count').textContent = pendingRows.length;
    pillPending.hidden = false;
  } else {
    pillPending.hidden = true;
  }
  let conflictCount = 0;
  for (const row of pendingRows) {
    if (isRowEdited(row)) conflictCount += 1;
  }
  const pillConflict = document.getElementById('pill-conflict');
  if (conflictCount > 0) {
    document.getElementById('pill-conflict-count').textContent = conflictCount;
    pillConflict.hidden = false;
  } else {
    pillConflict.hidden = true;
  }

  // Кнопка «Принять все» — disabled когда принимать нечего.
  document.getElementById('btn-accept-all').disabled = pendingRows.length === 0;

  // Body
  const body = document.getElementById('attr-table-body');
  body.innerHTML = '';
  for (const row of allRows) {
    body.appendChild(renderRow(row));
  }

  document.getElementById('product-card').hidden = false;
  document.getElementById('loading').hidden = true;
}

function renderRow(row) {
  const li = document.createElement('li');
  li.className = 'attr-row';
  if (row.kind === 'pending') {
    li.classList.add('row-pending');
    if (isRowEdited(row)) li.classList.add('row-edited');
  }

  // col 1: attr name (Russian) + snake_case sub
  const nameCell = document.createElement('div');
  const main = document.createElement('span');
  main.className = 'attr-name';
  main.textContent = attrLabel(row.attr);
  nameCell.appendChild(main);
  const sub = document.createElement('span');
  sub.className = 'attr-name-sub';
  sub.textContent = row.attr;
  nameCell.appendChild(sub);
  li.appendChild(nameCell);

  // col 2: value (text or inline-edit input for pending)
  const valueCell = document.createElement('div');
  valueCell.className = 'attr-value';
  if (row.kind === 'pending') {
    const currentVal = (state.edited[row.attr] !== undefined)
      ? state.edited[row.attr]
      : (row.pred.value ?? '');
    const inp = buildInput(state.preset, row.attr, currentVal);
    inp.classList.add('attr-value-input');
    const isEdited = li.classList.contains('row-edited');
    if (isEdited) inp.classList.add('is-edited');
    inp.addEventListener('input', () => { state.edited[row.attr] = inp.value; render(); });
    inp.addEventListener('change', () => { state.edited[row.attr] = inp.value; render(); });
    valueCell.appendChild(inp);
    if (isEdited) {
      const hint = document.createElement('div');
      hint.className = 'attr-value-hint';
      hint.innerHTML = `≠ система предлагала: <span class="system-value">${formatValue(row.attr, row.pred.value)}</span>`;
      valueCell.appendChild(hint);
    } else {
      const pmi = pmiTimesPhrase(row.pred.validation);
      if (pmi) {
        const hint = document.createElement('div');
        hint.className = 'attr-value-hint';
        hint.textContent =
          `комбинация ${attrLabel(pmi.attr)}=${formatValue(pmi.attr, pmi.value)} `
          + `+ ${attrLabel(row.attr)}=${formatValue(row.attr, row.pred.value)} `
          + `встречается в ×${pmi.times} реже обычного — проверьте`;
        valueCell.appendChild(hint);
      }
    }
  } else {
    valueCell.textContent = formatValue(row.attr, row.value);
  }
  li.appendChild(valueCell);

  // col 3: source layer pill
  const srcCell = document.createElement('div');
  srcCell.className = 'attr-source';
  if (row.kind === 'card') {
    srcCell.appendChild(makeLayerPill('from_card'));
  } else if (row.kind === 'auto') {
    srcCell.appendChild(makeLayerPill(row.layer));
  } else {
    srcCell.appendChild(makeLayerPill(row.pred.layer));
  }
  li.appendChild(srcCell);

  // col 4: confidence
  const confCell = document.createElement('div');
  confCell.className = 'attr-conf';
  if (row.kind === 'card') {
    const dash = document.createElement('span');
    dash.className = 'dash'; dash.textContent = '—';
    confCell.appendChild(dash);
  } else {
    const conf = row.kind === 'auto' ? row.confidence : row.pred.confidence;
    if (conf > 0) {
      const num = document.createElement('span');
      num.className = 'conf-num'; num.textContent = conf.toFixed(2);
      confCell.appendChild(num);
      const bar = document.createElement('span');
      bar.className = 'conf-bar';
      const fill = document.createElement('span');
      fill.style.width = `${Math.round(conf * 100)}%`;
      bar.appendChild(fill);
      confCell.appendChild(bar);
      if (conf < AUTO_ACCEPT) confCell.classList.add('low');
    } else {
      const dash = document.createElement('span');
      dash.className = 'dash'; dash.textContent = '—';
      confCell.appendChild(dash);
    }
  }
  li.appendChild(confCell);

  // col 5: check / actions
  const checkCell = document.createElement('div');
  checkCell.className = 'attr-check';
  if (row.kind === 'card') {
    const dash = document.createElement('span');
    dash.className = 'dash-cell'; dash.textContent = '—';
    checkCell.appendChild(dash);
  } else if (row.kind === 'auto') {
    const pill = document.createElement('span');
    pill.className = 'check-pill';
    const icon = document.createElement('span');
    icon.className = 'check-icon'; icon.textContent = '✓';
    pill.appendChild(icon);
    const label = document.createElement('span');
    label.innerHTML = 'принято <span class="check-sub">авто</span>';
    pill.appendChild(label);
    checkCell.appendChild(pill);
  } else {
    // pending — accept/reject
    const actions = document.createElement('div');
    actions.className = 'check-actions';
    const isEdited = li.classList.contains('row-edited');
    const accept = document.createElement('button');
    accept.type = 'button';
    accept.className = 'btn-accept';
    accept.textContent = isEdited ? 'Сохранить' : '✓ Принять';
    const currentVal = (state.edited[row.attr] !== undefined)
      ? state.edited[row.attr]
      : (row.pred.value ?? '');
    accept.disabled = (currentVal == null || currentVal === '');
    accept.addEventListener('click', () => acceptAttr(row.attr));
    actions.appendChild(accept);
    if (isEdited) {
      const revert = document.createElement('button');
      revert.type = 'button';
      revert.className = 'btn-revert';
      revert.title = 'Откатить к предложению системы';
      revert.textContent = '↺';
      revert.addEventListener('click', () => {
        delete state.edited[row.attr];
        render();
      });
      actions.appendChild(revert);
    } else {
      const reject = document.createElement('button');
      reject.type = 'button';
      reject.className = 'btn-reject';
      reject.title = 'Отклонить';
      reject.textContent = '✕';
      reject.addEventListener('click', () => rejectAttr(row.attr));
      actions.appendChild(reject);
    }
    checkCell.appendChild(actions);
  }
  li.appendChild(checkCell);

  return li;
}

function buildInput(presetName, attr, value) {
  const info = (ATTR_INFO[presetName] || {})[attr];
  const kind = info ? info.kind : null;

  if (kind === 'enum' || kind === 'bool') {
    const sel = document.createElement('select');
    const blank = document.createElement('option');
    blank.value = '';
    blank.textContent = '— не задано —';
    sel.appendChild(blank);
    info.states.forEach(s => {
      const opt = document.createElement('option');
      opt.value = s;
      opt.textContent = formatValue(attr, s);
      if (String(value) === s) opt.selected = true;
      sel.appendChild(opt);
    });
    return sel;
  }

  const inp = document.createElement('input');
  inp.type = 'text';
  inp.value = value != null ? String(value) : '';
  if (kind === 'numeric_bins' && info.states.length) {
    inp.placeholder = `число (бакеты: ${info.states.join(', ')})`;
  } else {
    inp.placeholder = '— не задано —';
  }
  return inp;
}

// ---------- Actions ----------

function acceptAttr(attr) {
  if (!state.response) {
    // Если каскад ещё не запускался, оператор всё равно может «принять» —
    // его значение сохраняется в confirmed, пойдёт на следующий «Дозаполнить».
    const val = state.edited[attr];
    if (val === null || val === undefined || val === '') return;
    state.confirmed[attr] = val;
    delete state.edited[attr];
    state.rejected.delete(attr);
    // Имитируем «уже зафиксированный» атрибут, чтобы строка переехала в auto.
    state.response = state.response || { predictions: {}, expected: {} };
    state.response.predictions[attr] = {
      value: val, layer: 'operator', confidence: 1.0, validation: null,
    };
    render();
    return;
  }
  const pred = state.response.predictions[attr];
  const val = (state.edited[attr] !== undefined && state.edited[attr] !== '')
            ? state.edited[attr]
            : pred?.value;
  if (val === null || val === undefined || val === '') return;
  state.confirmed[attr] = val;
  delete state.edited[attr];
  state.rejected.delete(attr);
  state.response.predictions[attr] = {
    value: val, layer: 'operator', confidence: 1.0, validation: null,
  };
  render();
}

function rejectAttr(attr) {
  state.rejected.add(attr);
  delete state.edited[attr];
  delete state.confirmed[attr];
  render();
}

function acceptAll() {
  if (!state.response) return;
  for (const attr of state.pendingAttrs) {
    if (state.rejected.has(attr)) continue;
    const pred = state.response.predictions[attr];
    const val = (state.edited[attr] !== undefined && state.edited[attr] !== '')
              ? state.edited[attr]
              : pred?.value;
    if (val === null || val === undefined || val === '') continue;
    state.confirmed[attr] = val;
    delete state.edited[attr];
    state.response.predictions[attr] = {
      value: val, layer: 'operator', confidence: 1.0, validation: null,
    };
  }
  render();
}

async function rerun() {
  // Главное действие: запускает каскад с накопленными confirmed как evidence.
  // Подтверждённые атрибуты приходят со слоем "operator", остальные
  // пересчитываются (LLM-fallback может выдать новые предложения).
  const btn = document.getElementById('btn-rerun');
  btn.disabled = true;
  const orig = btn.textContent;
  btn.textContent = 'Считаю…';
  try {
    const resp = await callEnrich(state.product, state.confirmed);
    state.response = resp;
    state.edited = {};
    render();
  } catch (e) {
    showError(e.message || String(e));
  } finally {
    btn.disabled = false;
    btn.textContent = orig;
  }
}

function showError(msg) {
  document.getElementById('error-text').textContent = msg;
  document.getElementById('error-card').hidden = false;
  document.getElementById('loading').hidden = true;
}

function loadPreset(name) {
  // ВАЖНО: каскад не запускается. До нажатия «Дозаполнить» карточка
  // показывает только partner-fields + категорию из preset, ML-атрибуты
  // висят как пустые pending.
  state.preset = name;
  state.product = PRESETS[name];
  state.confirmed = {};
  state.rejected = new Set();
  state.edited = {};
  state.response = null;
  document.getElementById('error-card').hidden = true;
  document.getElementById('loading').hidden = true;
  setHeroImage(state.product.image_url);
  render();
}

function setHeroImage(url) {
  const photo = document.getElementById('hero-photo');
  if (!photo) return;
  // placeholder с диагональными полосами — fallback, используется когда
  // url пустой или картинка не подгрузилась. Real-prod: тут будет img из CDN.
  photo.classList.add('photo-placeholder');
  photo.innerHTML = '<span>фото товара</span>';
  if (!url) return;
  const img = new Image();
  img.alt = state.product.product_name;
  img.onload = () => {
    photo.classList.remove('photo-placeholder');
    photo.innerHTML = '';
    photo.appendChild(img);
  };
  img.onerror = () => { /* остаётся placeholder */ };
  img.src = url;
}

// ---------- Bootstrap ----------

document.getElementById('preset-select').addEventListener('change', (e) => {
  loadPreset(e.target.value);
});
document.getElementById('btn-accept-all').addEventListener('click', acceptAll);
document.getElementById('btn-rerun').addEventListener('click', rerun);

(async () => {
  await loadCategoriesMetadata();
  loadPreset('chocolate');
})();
