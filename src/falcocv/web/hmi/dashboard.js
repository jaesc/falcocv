async function loadProjects() {
  try {
    const res = await fetch('/api/server/projects');
    const data = await res.json();
    render(data);
  } catch {
    document.getElementById('status').textContent = 'Failed to connect';
  }
}

function render(projects) {
  const tbody = document.getElementById('projects');
  const empty = document.getElementById('empty');
  const status = document.getElementById('status');

  tbody.innerHTML = '';
  if (projects.length === 0) {
    empty.style.display = 'block';
    status.textContent = 'No projects running';
    return;
  }
  empty.style.display = 'none';
  status.textContent = `${projects.length} project(s) — auto-refresh 3s`;

  for (const p of projects) {
    const tr = document.createElement('tr');
    const idCell = document.createElement('td');
    idCell.className = p.alive ? 'alive' : 'stopped';
    idCell.textContent = p.project_id;
    tr.appendChild(idCell);

    const pidCell = document.createElement('td');
    pidCell.textContent = p.alive ? String(p.pid) : '—';
    pidCell.className = p.alive ? '' : 'stopped';
    tr.appendChild(pidCell);

    const memCell = document.createElement('td');
    memCell.textContent = p.alive ? `${p.memory_mb} MB` : '—';
    memCell.className = p.alive ? '' : 'stopped';
    tr.appendChild(memCell);

    const uptCell = document.createElement('td');
    uptCell.textContent = p.alive ? fmtDuration(p.uptime_seconds) : '—';
    uptCell.className = p.alive ? '' : 'stopped';
    tr.appendChild(uptCell);

    const gridCell = document.createElement('td');
    gridCell.textContent = p.alive ? `${p.layout_rows}\u00d7${p.layout_cols}` : '—';
    tr.appendChild(gridCell);

    const slotCell = document.createElement('td');
    slotCell.textContent = String(p.slot_count);
    tr.appendChild(slotCell);

    const actCell = document.createElement('td');
    actCell.style.textAlign = 'right';
    if (p.alive) {
      const openBtn = document.createElement('button');
      openBtn.className = 'btn btn-open';
      openBtn.textContent = 'Open';
      openBtn.onclick = () => window.open(`/hmi?id=${p.project_id}`, '_blank');
      actCell.appendChild(openBtn);
      actCell.appendChild(document.createTextNode(' '));
    }
    const reloadBtn = document.createElement('button');
    reloadBtn.className = 'btn btn-reload';
    reloadBtn.textContent = 'Reload';
    reloadBtn.onclick = async () => {
      try {
        await fetch('/api/hmi/reload', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({project_id: p.project_id}),
        });
      } catch {}
    };
    actCell.appendChild(reloadBtn);
    if (p.alive) {
      actCell.appendChild(document.createTextNode(' '));
      const stopBtn = document.createElement('button');
      stopBtn.className = 'btn btn-stop';
      stopBtn.textContent = 'Stop';
      stopBtn.onclick = async () => {
        try {
          await fetch('/api/hmi/stop', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({project_id: p.project_id}),
          });
        } catch {}
      };
      actCell.appendChild(stopBtn);
    }
    tr.appendChild(actCell);
    tbody.appendChild(tr);
  }
}

function fmtDuration(s) {
  if (s < 60) return `${s.toFixed(0)}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${Math.floor(s % 60)}s`;
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return `${h}h ${m}m`;
}

loadProjects();
setInterval(loadProjects, 3000);
