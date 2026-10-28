const BASE = '';
const PROJECT_ID = new URLSearchParams(location.search).get('id') || 'default';
let ws = null;
let hmiData = null;
let imageCache = {};
let wsReconnectTimer = null;
let logEntries = [];
let logSince = 0;
let logCellIds = [];
let slotWidgetMap = {};
let _slotFetchControllers = {};

function getGridTrack(count, ratios) {
  if (ratios && ratios.length === count) {
    return ratios.map(r => `${r}fr`).join(' ');
  }
  return `repeat(${count}, 1fr)`;
}

function renderGrid(layout) {
  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  logCellIds = [];
  slotWidgetMap = {};

  const rows = layout.rows || 50;
  const cols = layout.cols || 50;
  const widgets = layout.widgets || [];

  grid.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
  grid.style.gridTemplateRows = `repeat(${rows}, 1fr)`;

  const slotStatusMap = {};
  for (const s of (hmiData.slots || [])) {
    slotStatusMap[s.name] = s.status || 'idle';
  }

  widgets.sort((a, b) => (a.z || 0) - (b.z || 0));
  for (const w of widgets) {
    const wt = w.type || 'None';
    const sx = (w.x || 0) + 1;
    const sy = (w.y || 0) + 1;
    const sw = w.w || 4;
    const sh = w.h || 3;
    const z = w.z || 0;
    const slotName = w.slot || '';

    const div = document.createElement('div');
    div.className = 'cell widget-' + wt.toLowerCase();
    div.style.gridColumn = `${sx} / span ${sw}`;
    div.style.gridRow = `${sy} / span ${sh}`;
    div.style.zIndex = z;

    if (w.label) {
      const lbl = document.createElement('div');
      lbl.className = 'cell-label';
      lbl.textContent = w.label;
      div.appendChild(lbl);
    }

    renderWidgetCell(div, wt, w, slotStatusMap, slotName);

    if (wt !== 'None') {
      const ct = document.createElement('div');
      ct.className = 'cell-type';
      ct.textContent = wt;
      div.appendChild(ct);
    }

    grid.appendChild(div);
  }

  flushLogCells();
  updateAllViewers();
}

function renderWidgetCell(container, wt, w, slotStatusMap, slotName) {
  if (wt === 'ImageViewer') {
    const vdiv = document.createElement('div');
    vdiv.className = 'viewer-cell';
    vdiv.id = `viewer-${w.id}`;
    const empty = document.createElement('span');
    empty.className = 'empty';
    empty.textContent = 'No image';
    vdiv.appendChild(empty);
    container.appendChild(vdiv);
  } else if (wt === 'MeasureLabel') {
    const mdiv = document.createElement('div');
    mdiv.className = 'measure-cell';
    mdiv.id = `measure-${w.id}`;
    const mfs = w.font_size || 14;
    const ha = w.halign || 'center';
    const ll = w.label_layout || 'stacked';
    mdiv.style.justifyContent = ha === 'left' ? 'flex-start' : ha === 'right' ? 'flex-end' : 'center';
    if (ll === 'inline') {
      mdiv.style.flexDirection = 'row';
      mdiv.style.alignItems = 'center';
    }
    if (w.label) {
      const mlbl = document.createElement('span');
      mlbl.className = 'measure-label';
      mlbl.style.fontSize = `${Math.max(8, mfs - 4)}px`;
      mlbl.textContent = ll === 'inline' ? (w.label + ':') : w.label;
      mdiv.appendChild(mlbl);
    }
    const mval = document.createElement('span');
    mval.className = 'measure-value';
    mval.id = `mv-${w.id}`;
    mval.style.fontSize = `${mfs}px`;
    mval.textContent = '\u2014';
    mdiv.appendChild(mval);
    container.appendChild(mdiv);
  } else if (wt === 'StatusIndicator') {
    const st = slotStatusMap[slotName] || 'idle';
    const sdiv = document.createElement('div');
    sdiv.className = 'indicator-cell';
    sdiv.id = `indicator-${w.id}`;
    const siColor = st === 'pass' ? (w.color_pass || '#2e7d32')
      : st === 'fail' ? (w.color_fail || '#c62828')
      : st === 'pending' ? (w.color_pending || '#c1c100')
      : (w.color_idle || '#888');
    sdiv.style.backgroundColor = siColor;
    if (st === 'pass') { sdiv.textContent = 'PASS'; }
    else if (st === 'fail') { sdiv.textContent = 'FAIL'; }
    else if (st === 'pending') { sdiv.textContent = 'PENDING'; }
    else { sdiv.textContent = 'IDLE'; }
    sdiv.style.color = '#fff';
    if (w.label) {
      const ilbl = document.createElement('div');
      ilbl.className = 'indicator-label';
      ilbl.textContent = w.label;
      sdiv.appendChild(ilbl);
    }
    container.appendChild(sdiv);
    if (slotName) {
      if (!slotWidgetMap[slotName]) slotWidgetMap[slotName] = [];
      slotWidgetMap[slotName].push({ 
        type: 'StatusIndicator', id: `indicator-${w.id}`,
        color_pass: w.color_pass, color_fail: w.color_fail,
        color_pending: w.color_pending, color_idle: w.color_idle
      });
    }
  } else if (wt === 'LedIndicator') {
    const st = slotStatusMap[slotName] || 'idle';
    const lwrap = document.createElement('div');
    lwrap.style.cssText = 'position:absolute;inset:8px;display:flex;flex-direction:column;align-items:center;justify-content:center;';
    const ldiv = document.createElement('div');
    ldiv.className = 'led-indicator';
    ldiv.id = `led-${w.id}`;
    const statusColor = st === 'pass' ? (w.color_pass || '#2e7d32')
      : st === 'fail' ? (w.color_fail || '#c62828')
      : st === 'pending' ? (w.color_pending || '#c1c100')
      : (w.color_idle || '#888');
    ldiv.style.backgroundColor = statusColor;
    ldiv.style.flex = '1';
    ldiv.style.aspectRatio = '1';
    ldiv.style.maxWidth = '100%';
    ldiv.style.maxHeight = '100%';
    ldiv.style.alignSelf = 'center';
    if (st === 'pending') ldiv.classList.add('pending');
    lwrap.appendChild(ldiv);
    if (w.label) {
      const llbl = document.createElement('div');
      llbl.className = 'led-label';
      llbl.textContent = w.label;
      lwrap.appendChild(llbl);
    }
    container.appendChild(lwrap);
    if (slotName) {
      if (!slotWidgetMap[slotName]) slotWidgetMap[slotName] = [];
      slotWidgetMap[slotName].push({ 
        type: 'LedIndicator', id: `led-${w.id}`,
        color_pass: w.color_pass, color_fail: w.color_fail,
        color_pending: w.color_pending, color_idle: w.color_idle
      });
    }
  } else if (wt === 'TextLabel') {
    const tdiv = document.createElement('div');
    tdiv.className = 'text-cell';
    tdiv.textContent = w.label || 'Label';
    const bg = w.bg_color || [45, 45, 45];
    const tc = w.text_color || [204, 204, 204];
    const fs = w.font_size || 14;
    const ha = w.halign || 'center';
    const va = w.valign || 'center';
    tdiv.style.backgroundColor = `rgb(${bg[0]},${bg[1]},${bg[2]})`;
    tdiv.style.color = `rgb(${tc[0]},${tc[1]},${tc[2]})`;
    tdiv.style.fontSize = `${fs}px`;
    tdiv.style.justifyContent = ha === 'left' ? 'flex-start' : ha === 'right' ? 'flex-end' : 'center';
    tdiv.style.alignItems = va === 'top' ? 'flex-start' : va === 'bottom' ? 'flex-end' : 'center';
    container.appendChild(tdiv);
  } else if (wt === 'LogPanel') {
    const ldiv = document.createElement('div');
    ldiv.className = 'log-cell';
    ldiv.id = `log-${w.id}`;
    const lfs = w.font_size || 14;
    ldiv.style.fontSize = `${lfs}px`;
    container.appendChild(ldiv);
    logCellIds.push(`log-${w.id}`);
  }
}

async function init() {
  await loadHMI();
  await fetchLogs();
  connectWS();
}

async function loadHMI() {
  try {
    const res = await fetch(`${BASE}/api/hmi?project_id=${PROJECT_ID}`);
    hmiData = await res.json();
    const hasLayout = (hmiData.layout?.rows || 0) > 0;
    document.getElementById('splash').classList.toggle('hidden', hasLayout);
    if (hasLayout) {
      renderGrid(hmiData.layout);
      updateFooter(hmiData.slots);
    }
  } catch (e) {
    document.getElementById('grid').innerHTML = '<div style="padding:40px;text-align:center;color:#555;font:14px sans-serif;">Failed to load HMI. Upload a .hmi file.</div>';
  }
}

async function fetchLogs() {
  try {
    const res = await fetch(`${BASE}/api/hmi/logs?project_id=${PROJECT_ID}&since=0`);
    const data = await res.json();
    if (data.logs && data.logs.length > 0) {
      logEntries = data.logs;
      logSince = data.count;
      flushLogCells();
    }
  } catch (e) { /* ignore */ }
}

function flushLogCells() {
  for (const cellId of logCellIds) {
    const el = document.getElementById(cellId);
    if (!el) continue;
    el.innerHTML = '';
    for (const entry of logEntries) {
      const line = document.createElement('div');
      line.className = `log-${entry.level || 'info'}`;
      line.textContent = `[${entry.timestamp}] ${entry.message}`;
      el.appendChild(line);
    }
    el.scrollTop = el.scrollHeight;
  }
}

function appendLog(entry) {
  logEntries.push(entry);
  const lineHtml = `<div class="log-${entry.level || 'info'}">[${entry.timestamp}] ${entry.message}</div>`;
  for (const cellId of logCellIds) {
    const el = document.getElementById(cellId);
    if (!el) continue;
    el.insertAdjacentHTML('beforeend', lineHtml);
    el.scrollTop = el.scrollHeight;
  }
}

function updateSlotStatus(slotName, status) {
  if (!slotWidgetMap[slotName]) return;
  for (const w of slotWidgetMap[slotName]) {
    const el = document.getElementById(w.id);
    if (!el) continue;
    if (w.type === 'LedIndicator') {
      const colors = {
        pass: w.color_pass || '#2e7d32',
        fail: w.color_fail || '#c62828',
        pending: w.color_pending || '#c1c100',
        idle: w.color_idle || '#888'
      };
      el.style.backgroundColor = colors[status] || colors.idle;
      el.classList.toggle('pending', status === 'pending');
    } else if (w.type === 'StatusIndicator') {
      const colors = {
        pass: w.color_pass || '#2e7d32',
        fail: w.color_fail || '#c62828',
        pending: w.color_pending || '#c1c100',
        idle: w.color_idle || '#888'
      };
      el.style.backgroundColor = colors[status] || colors.idle;
      el.classList.toggle('pending', status === 'pending');
      if (status === 'pass') { el.textContent = 'PASS'; }
      else if (status === 'fail') { el.textContent = 'FAIL'; }
      else if (status === 'pending') { el.textContent = 'PENDING'; }
      else { el.textContent = 'IDLE'; }
    } else {
      el.classList.remove('pass', 'fail', 'pending');
      if (status === 'pass') { el.classList.add('pass'); el.textContent = 'PASS'; }
      else if (status === 'fail') { el.classList.add('fail'); el.textContent = 'FAIL'; }
      else if (status === 'pending') { el.classList.add('pending'); el.textContent = 'PENDING'; }
      else { el.textContent = 'IDLE'; }
    }
  }
}

function updateFooter(slots) {
  const info = document.getElementById('slot-info');
  if (!slots || slots.length === 0) {
    info.textContent = 'No slots loaded';
    return;
  }
  info.textContent = slots.map(s => `${s.name}: ${s.state}${s.last_error ? ' ERR' : ''}`).join(' | ');
}

function updateAllViewers() {
  for (const w of (hmiData.layout?.widgets || [])) {
    if (w.slot && w.output && w.type === 'ImageViewer') {
      fetchOutput(w.slot, w.output);
    } else if (w.slot && w.output && w.type === 'MeasureLabel') {
      fetchOutput(w.slot, w.output, null, w.field, w.id);
    }
  }
}

async function fetchOutput(slotName, outputName, signal, field, widgetId) {
  try {
    let url = `${BASE}/api/hmi/slot/${encodeURIComponent(slotName)}/output/${encodeURIComponent(outputName)}?project_id=${PROJECT_ID}&quality=85`;
    if (field) url += `&field=${encodeURIComponent(field)}`;
    if (widgetId) url += `&widget_id=${encodeURIComponent(widgetId)}`;
    const res = await fetch(url, { signal });
    if (!res.ok) return;
    const data = await res.json();
    if (data.data_b64) {
      const key = `${slotName}-${outputName}`;
      imageCache[key] = {
        b64: `data:image/jpeg;base64,${data.data_b64}`,
        overlays: data.overlays || [],
        width: data.width,
        height: data.height
      };
      updateImageViewers(slotName, outputName);
    } else if (data.value !== undefined) {
      updateMeasureLabels(slotName, outputName, data);
    }
  } catch (e) {
    if (e.name === 'AbortError') return;
  }
}

function updateMeasureLabels(slotName, outputName, data) {
  const value = data.value;
  const status = data.status;
  const layout = hmiData.layout;
  for (const w of (layout.widgets || [])) {
    if (w.type === 'MeasureLabel' && w.slot === slotName && w.output === outputName) {
      const el = document.getElementById(`mv-${w.id}`);
      if (!el) continue;
      el.textContent = String(value);
      el.classList.remove('pass', 'fail');
      if (status === 'pass') el.classList.add('pass');
      else if (status === 'fail') el.classList.add('fail');
    }
  }
}

function renderOverlays(ctx, overlays, scale, ox, oy) {
  if (!overlays || !overlays.length) return;
  for (const elem of overlays) {
    if (elem.visible === false) continue;
    const d = elem.data;
    const color = d.color || [0, 255, 0];
    const colorCss = `rgb(${color[2]},${color[1]},${color[0]})`;
    const thickness = d.thickness || 2;
    const fill = d.fill || false;
    ctx.lineWidth = Math.max(1, thickness);
    ctx.strokeStyle = colorCss;

    switch (elem.element_type) {
      case 'roi': {
        const roi = d.roi;
        const x = roi[0], y = roi[1], w = roi[2], h = roi[3];
        const angle = roi.length > 4 ? roi[4] : 0;
        const cx = (x + w / 2) * scale + ox;
        const cy = (y + h / 2) * scale + oy;
        ctx.save();
        ctx.translate(cx, cy);
        if (angle) ctx.rotate(angle * Math.PI / 180);
        ctx.strokeRect(-w / 2 * scale, -h / 2 * scale, w * scale, h * scale);
        if (fill) {
          ctx.fillStyle = colorCss;
          ctx.fillRect(-w / 2 * scale, -h / 2 * scale, w * scale, h * scale);
        }
        ctx.restore();
        break;
      }
      case 'contour':
      case 'polygon': {
        const pts = d.contour || d.points || [];
        if (pts.length < 2) break;
        const isNested = Array.isArray(pts[0][0]);
        const px = (i) => (isNested ? pts[i][0][0] : pts[i][0]) * scale + ox;
        const py = (i) => (isNested ? pts[i][0][1] : pts[i][1]) * scale + oy;
        ctx.beginPath();
        ctx.moveTo(px(0), py(0));
        for (let i = 1; i < pts.length; i++) {
          ctx.lineTo(px(i), py(i));
        }
        ctx.closePath();
        if (fill) {
          ctx.fillStyle = colorCss;
          ctx.fill();
        }
        ctx.stroke();
        break;
      }
      case 'circle': {
        const cx = d.center[0] * scale + ox;
        const cy = d.center[1] * scale + oy;
        const r = d.radius * scale;
        ctx.beginPath();
        ctx.arc(cx, cy, Math.max(1, r), 0, 2 * Math.PI);
        if (fill) {
          ctx.fillStyle = colorCss;
          ctx.fill();
        }
        ctx.stroke();
        break;
      }
      case 'line': {
        const sx = d.start[0] * scale + ox;
        const sy = d.start[1] * scale + oy;
        const ex = d.end[0] * scale + ox;
        const ey = d.end[1] * scale + oy;
        ctx.beginPath();
        ctx.moveTo(sx, sy);
        ctx.lineTo(ex, ey);
        ctx.stroke();
        break;
      }
      case 'text': {
        const tx = d.position[0] * scale + ox;
        const ty = d.position[1] * scale + oy;
        const fs = d.font_scale || 0.5;
        ctx.font = `${Math.max(8, fs * 20)}px Consolas`;
        ctx.fillStyle = colorCss;
        ctx.fillText(String(d.text), tx, ty);
        break;
      }
      case 'point': {
        const px = d.center[0] * scale + ox;
        const py = d.center[1] * scale + oy;
        const r = Math.max(1, (d.radius || 4) / 2);
        ctx.lineWidth = Math.max(1, 2);
        ctx.beginPath();
        ctx.moveTo(px - r, py);
        ctx.lineTo(px + r, py);
        ctx.moveTo(px, py - r);
        ctx.lineTo(px, py + r);
        ctx.stroke();
        break;
      }
      case 'axes': {
        const acx = d.center[0] * scale + ox;
        const acy = d.center[1] * scale + oy;
        const aLen = (d.length || 30) * scale;
        const aAngle = (d.angle || 0) * Math.PI / 180;
        const colorX = d.color_x || [0, 0, 255];
        const colorY = d.color_y || [0, 255, 0];
        const aThick = d.thickness || 2;
        const ex1 = acx + aLen * Math.cos(aAngle);
        const ey1 = acy + aLen * Math.sin(aAngle);
        const perpAngle = aAngle + Math.PI / 2;
        const ex2 = acx + aLen * Math.cos(perpAngle);
        const ey2 = acy + aLen * Math.sin(perpAngle);
        ctx.lineWidth = Math.max(1, aThick);
        ctx.strokeStyle = `rgb(${colorX[2]},${colorX[1]},${colorX[0]})`;
        ctx.beginPath();
        ctx.moveTo(acx, acy);
        ctx.lineTo(ex1, ey1);
        ctx.stroke();
        ctx.strokeStyle = `rgb(${colorY[2]},${colorY[1]},${colorY[0]})`;
        ctx.beginPath();
        ctx.moveTo(acx, acy);
        ctx.lineTo(ex2, ey2);
        ctx.stroke();
        break;
      }
    }
  }
}

function updateImageViewers(slotName, outputName) {
  const layout = hmiData.layout;
  for (const w of (layout.widgets || [])) {
    if (w.type === 'ImageViewer' && w.slot === slotName && w.output === outputName) {
      const vdiv = document.getElementById(`viewer-${w.id}`);
      if (!vdiv) continue;
      const key = `${slotName}-${outputName}`;
      const cached = imageCache[key];
      if (cached && cached.b64) {
        const img = new Image();
        img.onload = () => {
          let canvas = vdiv.querySelector('canvas');
          if (!canvas) {
            vdiv.innerHTML = '';
            canvas = document.createElement('canvas');
            canvas.id = `canv-${w.id}`;
            vdiv.appendChild(canvas);
          }
          canvas.width = vdiv.clientWidth;
          canvas.height = vdiv.clientHeight;
          const ctx = canvas.getContext('2d');
          ctx.fillStyle = '#1a1a1a';
          ctx.fillRect(0, 0, canvas.width, canvas.height);
          const scale = Math.min(canvas.width / img.width, canvas.height / img.height);
          const ox = (canvas.width - img.width * scale) / 2;
          const oy = (canvas.height - img.height * scale) / 2;
          ctx.drawImage(img, ox, oy, img.width * scale, img.height * scale);
          renderOverlays(ctx, cached.overlays, scale, ox, oy);
        };
        img.src = cached.b64;
      }
    }
  }
}

function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${location.host}/api/ws?project_id=${PROJECT_ID}`);
  ws.onopen = () => {
    document.getElementById('ws-status').textContent = 'WS ●';
    document.getElementById('ws-status').style.color = '#a0ffa0';
  };
  ws.onclose = () => {
    document.getElementById('ws-status').textContent = 'WS ○';
    document.getElementById('ws-status').style.color = '#ffa0a0';
    wsReconnectTimer = setTimeout(connectWS, 3000);
  };
  ws.onmessage = (ev) => {
    try {
      const msg = JSON.parse(ev.data);
      if (msg.type === 'execution_finished') {
        refreshHMI();
      } else if (msg.type === 'hmi_log') {
        appendLog(msg.entry);
      } else if (msg.type === 'hmi_status') {
        updateSlotStatus(msg.slot, msg.status);
      } else if (msg.type === 'hmi_state_changed') {
        updateSlotState(msg.slot, msg.state);
      } else if (msg.type === 'hmi_slot_finished') {
        onSlotFinished(msg);
      } else if (msg.type === 'hmi_loaded') {
        location.reload();
      } else if (msg.type === 'hmi_stopped') {
        loadHMI();
      } else if (msg.type === 'hmi_prompt') {
        showPrompt(msg);
      } else if (msg.type === 'hmi_prompt_closed') {
        closePrompt(msg.req_id);
      }
    } catch (e) { /* ignore */ }
  };
}

function updateSlotState(slotName, state) {
  if (hmiData && hmiData.slots) {
    for (const s of hmiData.slots) {
      if (s.name === slotName) {
        s.state = state;
        break;
      }
    }
  }
  updateFooter(hmiData ? hmiData.slots : null);
}

function onSlotFinished(msg) {
  if (hmiData && hmiData.slots) {
    for (const s of hmiData.slots) {
      if (s.name === msg.slot) {
        s.state = 'idle';
        if (msg.error) s.last_error = msg.error;
        else s.last_error = null;
        break;
      }
    }
    updateFooter(hmiData.slots);
  }

  if (_slotFetchControllers[msg.slot]) {
    _slotFetchControllers[msg.slot].abort();
  }
  _slotFetchControllers[msg.slot] = new AbortController();
  const signal = _slotFetchControllers[msg.slot].signal;
  for (const w of (hmiData.layout?.widgets || [])) {
    if (w.slot === msg.slot && w.output && w.type === 'ImageViewer') {
      fetchOutput(w.slot, w.output, signal);
    } else if (w.slot === msg.slot && w.output && w.type === 'MeasureLabel') {
      fetchOutput(w.slot, w.output, signal, w.field, w.id);
    }
  }
}

async function refreshHMI() {
  await loadHMI();
  updateAllViewers();
}

function createPromptOverlay() {
  const overlay = document.createElement('div');
  overlay.id = 'prompt-overlay';
  overlay.innerHTML = `<div id="prompt-dialog">
    <div class="prompt-msg"></div>
    <div class="prompt-input"><input type="text" id="prompt-input-field"></div>
    <div class="prompt-buttons"></div>
  </div>`;
  overlay.querySelector('#prompt-input-field').onkeydown = (e) => {
    if (e.key === 'Enter') overlay.querySelector('#prompt-ok-btn')?.click();
  };
  document.body.appendChild(overlay);
  return overlay;
}

function showPrompt(msg) {
  let overlay = document.getElementById('prompt-overlay');
  if (!overlay) overlay = createPromptOverlay();
  overlay._prompt_req_id = msg.req_id;
  overlay.querySelector('.prompt-msg').textContent = msg.msg;
  const inputDiv = overlay.querySelector('.prompt-input');
  const btnDiv = overlay.querySelector('.prompt-buttons');
  inputDiv.style.display = 'none';
  btnDiv.style.display = 'none';

  const dialog = document.getElementById('prompt-dialog');
  if (msg.width) dialog.style.width = msg.width;
  if (msg.height) dialog.style.height = msg.height;
  if (msg.font_size) {
    dialog.style.fontSize = msg.font_size;
    dialog.querySelector('.prompt-msg').style.fontSize = msg.font_size;
    dialog.querySelector('#prompt-input-field').style.fontSize = msg.font_size;
  }
  if (msg.input_type === 'text') {
    inputDiv.style.display = 'block';
    const field = overlay.querySelector('#prompt-input-field');
    field.value = '';
    field.focus();
    const okBtn = document.createElement('button');
    okBtn.id = 'prompt-ok-btn';
    okBtn.textContent = 'OK';
    okBtn.onclick = () => respond(msg.req_id, field.value);
    btnDiv.style.display = 'flex';
    btnDiv.innerHTML = '';
    btnDiv.appendChild(okBtn);
  } else if (msg.input_type === 'confirm') {
    btnDiv.style.display = 'flex';
    btnDiv.innerHTML = '';
    for (const opt of (msg.options || ['Yes', 'No', 'Cancel'])) {
      const btn = document.createElement('button');
      btn.textContent = opt;
      btn.onclick = () => respond(msg.req_id, opt);
      btnDiv.appendChild(btn);
    }
  }
  overlay.style.display = 'flex';
}

function closePrompt(reqId) {
  const overlay = document.getElementById('prompt-overlay');
  if (overlay && overlay._prompt_req_id === reqId) {
    overlay.style.display = 'none';
  }
}

function respond(reqId, value) {
  if (ws) ws.send(JSON.stringify({cmd: 'prompt_response', req_id: reqId, value}));
  document.getElementById('prompt-overlay').style.display = 'none';
}

init();
