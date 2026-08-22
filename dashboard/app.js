// ─── State ────────────────────────────────────────────────────────────────────
const API = window.location.origin;
let currentRunId = null;
let trueExceptionIds = [];
let allAuditEntries = [];
let currentDemoMode = false;

const STAGE_LABELS = {
  level1_deterministic: 'L1 Deterministic',
  level2_deterministic_utr: 'L2 UTR Match',
  level2_rules_rounding: 'L2 Rounding',
  level2_rules_missing_utr: 'L2 Missing UTR',
  level2_rules_lag: 'L2 Settlement Lag',
  level2_rules_merged_batch: 'L2 Merged Batch',
  level2_ml_high_confidence: 'L2 ML (XGBoost)',
  level2_agent: 'L2 Agent (LLM) ⚡',
  level2_agent_fallback: 'L2 Agent (fallback)',
  level2_unresolved: 'L2 Unresolved',
  level2_pending: 'L2 Pending',
};

// ─── Tabs ──────────────────────────────────────────────────────────────────────
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById(`panel-${tab.dataset.tab}`).classList.add('active');
  });
});

// ─── Run buttons ───────────────────────────────────────────────────────────────
document.getElementById('runBtn').addEventListener('click', () => runReconciliation(false));
document.getElementById('runDemoBtn').addEventListener('click', () => runReconciliation(true));

// ─── Audit filters ─────────────────────────────────────────────────────────────
document.getElementById('filterStage').addEventListener('change', renderAudit);
document.getElementById('filterDecision').addEventListener('change', renderAudit);
document.getElementById('filterRecord').addEventListener('input', renderAudit);

// ─── Modal close ──────────────────────────────────────────────────────────────
document.getElementById('modalClose').addEventListener('click', closeModal);
document.getElementById('drillModal').addEventListener('click', e => {
  if (e.target === e.currentTarget) closeModal();
});
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });

// ─── Run pipeline ─────────────────────────────────────────────────────────────
async function runReconciliation(demoMode = false) {
  currentDemoMode = demoMode;
  const btn = demoMode ? document.getElementById('runDemoBtn') : document.getElementById('runBtn');
  const otherBtn = demoMode ? document.getElementById('runBtn') : document.getElementById('runDemoBtn');
  const status = document.getElementById('runStatus');

  [btn, otherBtn].forEach(b => { b.disabled = true; });
  btn.classList.add('loading');
  status.textContent = demoMode ? '⚡ Running demo mode (LLM will fire)…' : 'Running pipeline…';

  try {
    const runRes = await fetch(`${API}/run?demo_agent=${demoMode}`, { method: 'POST' });
    const { run_id } = await runRes.json();
    currentRunId = run_id;

    const [results, exceptions, audit] = await Promise.all([
      fetch(`${API}/results/${run_id}`).then(r => r.json()),
      fetch(`${API}/exceptions/${run_id}`).then(r => r.json()),
      fetch(`${API}/audit/${run_id}`).then(r => r.json()),
    ]);

    trueExceptionIds = exceptions.true_exception_ids;
    allAuditEntries = audit.entries;

    renderSummary(results);
    renderStageChart(results.stage_breakdown, demoMode);
    renderExceptions(exceptions.exceptions);
    populateStageFilter(results.stage_breakdown);
    renderAudit();

    // Auto-load evaluation
    loadEvaluation(run_id);

    const banner = document.getElementById('demoHelperBanner');
    if (banner) {
      const liveCalls = results.llm_call_count || 0;
      const fallbackCalls = results.agent_fallback_count || 0;
      let agentMsg = '0 LLM calls (rules resolved 100%)';
      if (liveCalls > 0) agentMsg = `${liveCalls} live LLM call(s) executed`;
      else if (fallbackCalls > 0) agentMsg = `${fallbackCalls} fallback call(s) (no API key)`;
      banner.innerHTML = `<span class="helper-icon">✓</span> <span><strong>Run ${run_id} complete:</strong> ${results.l1_matched}/${results.l1_total} orders &amp; ${results.l2_matched}/${results.l2_total} batches matched. ${agentMsg}.</span>`;
    }

    const llmMsg = results.llm_call_count > 0
      ? ` · ⚡ ${results.llm_call_count} LLM call(s) made`
      : '';
    status.textContent = `Run ${run_id} completed${llmMsg}`;

    // If demo mode, switch to audit tab and highlight the agent row
    if (demoMode && results.llm_call_count > 0) {
      setTimeout(() => {
        document.querySelector('[data-tab="audit"]').click();
        highlightAgentRow();
      }, 300);
    }
  } catch (err) {
    status.textContent = `Error: ${err.message}`;
    console.error(err);
  } finally {
    [btn, otherBtn].forEach(b => { b.disabled = false; });
    btn.classList.remove('loading');
  }
}

// ─── Summary cards ─────────────────────────────────────────────────────────────
function renderSummary(r) {
  document.getElementById('matchRateL1').textContent = `${(r.match_rate_l1 * 100).toFixed(1)}%`;
  document.getElementById('matchRateL1Sub').textContent = `${r.l1_matched}/${r.l1_total} orders ↔ settlement`;
  document.getElementById('matchRateL2').textContent = `${(r.match_rate_l2 * 100).toFixed(1)}%`;
  document.getElementById('matchRateL2Sub').textContent = `${r.l2_matched}/${r.l2_total} batches ↔ bank`;
  document.getElementById('excPrecision').textContent = `${(r.exception_precision * 100).toFixed(1)}%`;
  document.getElementById('excRecall').textContent = `recall ${(r.exception_recall * 100).toFixed(0)}%`;
  document.getElementById('llmCalls').textContent = r.llm_call_count;
  
  const fallbackCount = r.agent_fallback_count || 0;
  if (r.llm_call_count > 0 && fallbackCount === 0) {
    document.getElementById('llmCallsSub').textContent = `⚡ ${r.llm_call_count} live agent call(s)`;
  } else if (r.llm_call_count > 0 && fallbackCount > 0) {
    document.getElementById('llmCallsSub').textContent = `⚡ ${r.llm_call_count} live + ${fallbackCount} fallback`;
  } else if (r.llm_call_count === 0 && fallbackCount > 0) {
    document.getElementById('llmCallsSub').textContent = `0 live (+${fallbackCount} fallback, no API key)`;
  } else {
    document.getElementById('llmCallsSub').textContent = 'gray-zone only (none needed)';
  }
}

// ─── Stage chart ───────────────────────────────────────────────────────────────
function renderStageChart(breakdown, demoMode = false) {
  const wrap = document.getElementById('stageChartWrap');
  const entries = Object.entries(breakdown).filter(([, v]) => v > 0);
  if (!entries.length) { wrap.innerHTML = '<p style="color:var(--text-muted);padding:1rem">No data yet.</p>'; return; }
  const max = Math.max(...entries.map(([, v]) => v), 1);

  wrap.innerHTML = `<div class="bar-chart">${entries.map(([key, val]) => {
    const isAgent = key === 'level2_agent';
    return `<div class="bar-group">
      <div class="bar-value">${val}</div>
      <div class="bar ${isAgent ? 'bar-agent' : ''}" style="height:${(val / max) * 160}px" title="${STAGE_LABELS[key] || key}: ${val} records"></div>
      <div class="bar-label">${STAGE_LABELS[key] || key}</div>
    </div>`;
  }).join('')}</div>`;
}

// ─── Exception table ───────────────────────────────────────────────────────────
function renderExceptions(exceptions) {
  const tbody = document.querySelector('#exceptionTable tbody');
  tbody.innerHTML = exceptions.map(e => {
    const isTrue = trueExceptionIds.includes(e.record_id);
    const rowClass = isTrue ? 'row-true-exception' : 'row-false-exception';
    return `<tr class="${rowClass}">
      <td class="mono">${e.record_id}</td>
      <td>${e.side}</td>
      <td><span class="badge badge-exception">${e.reason_code}</span></td>
      <td>${e.confidence.toFixed(2)}</td>
      <td>${e.agent_rationale}</td>
    </tr>`;
  }).join('') || '<tr><td colspan="5" style="color:var(--text-muted);text-align:center;padding:1rem">No exceptions found.</td></tr>';
}

// ─── Audit table ───────────────────────────────────────────────────────────────
function populateStageFilter(breakdown) {
  const sel = document.getElementById('filterStage');
  sel.innerHTML = '<option value="">All stages</option>';
  Object.keys(breakdown).forEach(key => {
    sel.innerHTML += `<option value="${key}">${STAGE_LABELS[key] || key}</option>`;
  });
}

function renderAudit() {
  const stage = document.getElementById('filterStage').value;
  const decision = document.getElementById('filterDecision').value;
  const record = document.getElementById('filterRecord').value.toLowerCase();

  let entries = allAuditEntries;
  if (stage) entries = entries.filter(e => e.resolved_by === stage);
  if (decision) entries = entries.filter(e => e.decision === decision);
  if (record) entries = entries.filter(e => e.record_id.toLowerCase().includes(record));

  const tbody = document.querySelector('#auditTable tbody');
  tbody.innerHTML = entries.map(e => {
    const badgeClass = e.decision === 'MATCH' ? 'badge-match'
      : e.decision === 'EXCEPTION' ? 'badge-exception' : 'badge-borderline';
    const shapHtml = e.shap_json ? renderShapMini(e.shap_json) : '<span style="color:var(--text-dim)">—</span>';
    const isAgent = e.resolved_by === 'level2_agent';
    const rowStyle = isAgent ? 'background: rgba(245,158,11,0.04);' : '';
    return `<tr style="${rowStyle}" id="audit-row-${e.record_id}">
      <td class="mono">${e.record_id}</td>
      <td><span class="badge ${badgeClass}">${e.decision}</span></td>
      <td>${STAGE_LABELS[e.resolved_by] || e.resolved_by}${isAgent ? ' ⚡' : ''}</td>
      <td class="mono">${e.matched_to || '<span style="color:var(--text-dim)">—</span>'}</td>
      <td><span class="badge ${e.reason_code === 'CLEAN' ? 'badge-match' : 'badge-borderline'}">${e.reason_code || '—'}</span></td>
      <td style="max-width:220px;font-size:0.75rem;color:var(--text-muted)">${e.rationale || '—'}</td>
      <td>${shapHtml}</td>
      <td><button class="btn-drill" onclick="openDrill('${e.record_id}')">→ Drill</button></td>
    </tr>`;
  }).join('') || '<tr><td colspan="8" style="color:var(--text-muted);text-align:center;padding:1.5rem">Run reconciliation first.</td></tr>';
}

function highlightAgentRow() {
  const agentEntries = allAuditEntries.filter(e => e.resolved_by === 'level2_agent');
  if (!agentEntries.length) return;
  const rid = agentEntries[0].record_id;
  // Filter to show only agent rows
  document.getElementById('filterStage').value = 'level2_agent';
  renderAudit();
  // Scroll to it
  const row = document.getElementById(`audit-row-${rid}`);
  if (row) row.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// ─── SHAP mini bars ────────────────────────────────────────────────────────────
function renderShapMini(shapJson) {
  try {
    const shap = typeof shapJson === 'string' ? JSON.parse(shapJson) : shapJson;
    const vals = Object.values(shap).map(Math.abs);
    const max = Math.max(...vals, 0.001);
    return `<div class="shap-mini">${Object.entries(shap).map(([k, v]) => {
      const pct = Math.abs(v) / max * 100;
      const neg = v < 0;
      return `<div class="shap-bar-row">
        <span style="width:52px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${k.replace(/_/g,' ')}</span>
        <div class="shap-bar-track">
          <div class="shap-bar-fill ${neg ? 'negative' : ''}" style="width:${pct}%"></div>
        </div>
        <span style="width:36px;text-align:right">${v.toFixed(3)}</span>
      </div>`;
    }).join('')}</div>`;
  } catch { return '—'; }
}

function renderShapFull(shapJson) {
  if (!shapJson) return '';
  try {
    const shap = typeof shapJson === 'string' ? JSON.parse(shapJson) : shapJson;
    const vals = Object.values(shap).map(Math.abs);
    const max = Math.max(...vals, 0.001);
    return `<div class="drill-section-title">SHAP Feature Contributions</div>
    <div style="display:flex;flex-direction:column;gap:6px">${Object.entries(shap).map(([k, v]) => {
      const pct = Math.abs(v) / max * 100;
      const neg = v < 0;
      const color = neg ? 'var(--danger)' : 'var(--primary)';
      return `<div class="shap-bar-row">
        <span style="width:110px;font-size:0.75rem;color:var(--text-muted)">${k.replace(/_/g,' ')}</span>
        <div class="shap-bar-track" style="flex:1">
          <div class="shap-bar-fill ${neg ? 'negative' : ''}" style="width:${pct}%;background:${color}"></div>
        </div>
        <span style="width:50px;text-align:right;font-size:0.75rem;color:${color}">${v > 0 ? '+' : ''}${v.toFixed(4)}</span>
      </div>`;
    }).join('')}</div>`;
  } catch { return ''; }
}

// ─── Drill-down modal ──────────────────────────────────────────────────────────
function openDrill(recordId) {
  if (!currentRunId) return;
  const modal = document.getElementById('drillModal');
  const body = document.getElementById('modalBody');
  const title = document.getElementById('modalTitle');

  title.textContent = `Record: ${recordId}`;
  body.innerHTML = '<div class="loading-spinner">Loading record…</div>';
  modal.classList.add('open');
  document.body.style.overflow = 'hidden';

  fetch(`${API}/record/${currentRunId}/${encodeURIComponent(recordId)}`)
    .then(r => r.json())
    .then(data => { body.innerHTML = renderDrillContent(data); })
    .catch(err => { body.innerHTML = `<p style="color:var(--danger)">Error: ${err.message}</p>`; });
}

function closeModal() {
  document.getElementById('drillModal').classList.remove('open');
  document.body.style.overflow = '';
}

function renderDrillContent(data) {
  const { settlement, bank_row, audit_entry, merged_with } = data;
  const isAgent = audit_entry?.resolved_by === 'level2_agent';

  const fmt = (paise) => paise != null ? `₹${(paise / 100).toLocaleString('en-IN', {minimumFractionDigits: 2})}` : '—';

  let html = '<div class="drill-grid">';

  // Settlement card
  if (settlement) {
    html += `<div class="drill-card">
      <div class="drill-card-title">Settlement Batch</div>
      <div class="drill-field">
        <div class="drill-field-label">Settlement ID</div>
        <div class="drill-field-value">${settlement.settlement_id}</div>
      </div>
      <div class="drill-field">
        <div class="drill-field-label">Net Total</div>
        <div class="drill-field-value big">${fmt(settlement.net_total)}</div>
      </div>
      <div class="drill-field">
        <div class="drill-field-label">Settled Date</div>
        <div class="drill-field-value">${settlement.settled_date}</div>
      </div>
      <div class="drill-field">
        <div class="drill-field-label">Entity IDs (${settlement.entity_ids.length})</div>
        <div class="drill-field-value" style="font-size:0.7rem;color:var(--text-muted)">${settlement.entity_ids.slice(0,3).join(', ')}${settlement.entity_ids.length > 3 ? ` +${settlement.entity_ids.length-3} more` : ''}</div>
      </div>
    </div>`;
  } else {
    html += `<div class="drill-card" style="color:var(--text-dim)">
      <div class="drill-card-title">Settlement Batch</div>
      <p style="font-size:0.8125rem">No settlement batch found.</p>
    </div>`;
  }

  // Bank card
  if (bank_row) {
    html += `<div class="drill-card">
      <div class="drill-card-title">Bank Statement Row</div>
      <div class="drill-field">
        <div class="drill-field-label">Bank Row ID</div>
        <div class="drill-field-value">${bank_row.bank_row_id}</div>
      </div>
      <div class="drill-field">
        <div class="drill-field-label">Amount</div>
        <div class="drill-field-value big" style="color:var(--success)">${fmt(bank_row.amount)}</div>
      </div>
      <div class="drill-field">
        <div class="drill-field-label">Date</div>
        <div class="drill-field-value">${bank_row.date}</div>
      </div>
      <div class="drill-field">
        <div class="drill-field-label">UTR</div>
        <div class="drill-field-value">${bank_row.utr || '<span style="color:var(--warning)">MISSING</span>'}</div>
      </div>
      <div class="drill-field">
        <div class="drill-field-label">Narration</div>
        <div class="drill-field-value" style="font-size:0.7rem;color:var(--text-muted)">${bank_row.narration}</div>
      </div>
    </div>`;
  } else {
    html += `<div class="drill-card" style="color:var(--text-dim)">
      <div class="drill-card-title">Bank Statement Row</div>
      <p style="font-size:0.8125rem">No bank row matched — this is an exception.</p>
    </div>`;
  }

  html += '</div>';

  // Merged batch detail
  if (merged_with && merged_with.length > 0) {
    html += `<div class="drill-section-title">⊕ Merged With (MERGED_BATCH)</div>`;
    merged_with.forEach(m => {
      html += `<div class="drill-card" style="margin-bottom:0.5rem">
        <div class="drill-field"><div class="drill-field-label">Settlement ID</div><div class="drill-field-value">${m.settlement_id}</div></div>
        <div class="drill-field"><div class="drill-field-label">Net Total</div><div class="drill-field-value big" style="font-size:0.9rem">${fmt(m.net_total)}</div></div>
        <div class="drill-field"><div class="drill-field-label">Settled Date</div><div class="drill-field-value">${m.settled_date}</div></div>
      </div>`;
    });

    // Combined total confirmation
    if (settlement && bank_row) {
      const combined = settlement.net_total + merged_with.reduce((s, m) => s + m.net_total, 0);
      const diff = Math.abs(combined - bank_row.amount);
      html += `<div style="background:var(--success-bg);border:1px solid var(--success);border-radius:var(--radius-sm);padding:0.75rem;margin-bottom:1rem;font-size:0.8125rem">
        ✓ Combined: ${fmt(combined)} vs bank: ${fmt(bank_row.amount)} — diff: <strong>${diff} paise</strong>
      </div>`;
    }
  }

  // Decision section
  if (audit_entry) {
    const badgeClass = audit_entry.decision === 'MATCH' ? 'badge-match' : 'badge-exception';
    html += `<div class="drill-section-title">Pipeline Decision</div>
    <div style="display:flex;gap:0.75rem;align-items:center;margin-bottom:0.75rem;flex-wrap:wrap">
      <span class="badge ${badgeClass}" style="font-size:0.8125rem;padding:0.3rem 0.75rem">${audit_entry.decision}</span>
      <span style="color:var(--text-muted);font-size:0.8125rem">${STAGE_LABELS[audit_entry.resolved_by] || audit_entry.resolved_by}${isAgent ? ' ⚡' : ''}</span>
      <span style="background:var(--surface2);border:1px solid var(--border);border-radius:4px;padding:0.15rem 0.5rem;font-size:0.75rem">
        confidence: <strong>${audit_entry.confidence != null ? audit_entry.confidence.toFixed(3) : '—'}</strong>
      </span>
    </div>
    <div style="font-size:0.8125rem;color:var(--text-muted);margin-bottom:1rem;background:var(--surface2);padding:0.75rem;border-radius:var(--radius-sm);border:1px solid var(--border)">
      ${audit_entry.rationale || '—'}
    </div>`;

    // SHAP values
    if (audit_entry.shap_json) {
      html += renderShapFull(audit_entry.shap_json);
    }

    // Agent tool trace
    if (isAgent && audit_entry.tool_trace_json) {
      let trace = [];
      try { trace = JSON.parse(audit_entry.tool_trace_json); } catch {}
      if (trace.length > 0) {
        html += `<div class="agent-trace">
          <div class="agent-trace-header">⚡ Agent Tool Trace — ${trace.length} tool call(s)</div>
          ${trace.map((step, i) => `
          <details class="trace-step">
            <summary class="trace-step-summary">
              <span class="tool-name">${step.tool}()</span>
              <span style="color:var(--text-dim);font-size:0.7rem">${step.input ? JSON.stringify(step.input).slice(0, 60) + '…' : ''}</span>
            </summary>
            <div class="trace-step-body"><strong>Input:</strong>
${JSON.stringify(step.input, null, 2)}

<strong>Output:</strong>
${typeof step.output === 'string' ? step.output : JSON.stringify(step.output, null, 2)}</div>
          </details>`).join('')}
        </div>`;
      }
    }
  }

  return html;
}

// ─── Evaluation tab ────────────────────────────────────────────────────────────
async function loadEvaluation(runId) {
  try {
    const data = await fetch(`${API}/evaluation/${runId}`).then(r => r.json());
    renderEvalMetrics(data);
    renderEvalTables(data);
  } catch (err) {
    console.warn('Evaluation load failed:', err);
  }
}

function renderEvalMetrics(data) {
  const set = (id, val, barId, max = 1) => {
    const el = document.getElementById(id);
    const bar = document.getElementById(barId);
    if (el) el.textContent = typeof val === 'number' && max === 1
      ? `${(val * 100).toFixed(1)}%`
      : String(val);
    if (bar) bar.style.width = `${Math.min((typeof val === 'number' ? val : 0) / max * 100, 100)}%`;
  };

  set('pct-exc-prec', data.exception_precision, 'bar-exc-prec');
  set('pct-exc-rec',  data.exception_recall,    'bar-exc-rec');
  set('pct-l1',       data.match_rate_l1,        'bar-l1');
  set('pct-l2',       data.match_rate_l2,        'bar-l2');

  const fmCount = document.getElementById('pct-fm');
  const fmBar   = document.getElementById('bar-fm');
  if (fmCount) fmCount.textContent = data.false_matches.length;
  if (fmBar)   fmBar.style.width = data.false_matches.length > 0 ? '30%' : '0';

  const llmCount = document.getElementById('pct-llm');
  const llmBar   = document.getElementById('bar-llm');
  if (llmCount) llmCount.textContent = data.llm_call_count;
  if (llmBar)   llmBar.style.width = `${Math.min(data.llm_call_count * 20, 100)}%`;
}

function renderEvalTables(data) {
  const excRows  = data.rows.filter(r => r.category === 'true_exception');
  const resRows  = data.rows.filter(r => r.category === 'resolvable_challenge');

  // Exception table
  const excCorrect = excRows.filter(r => r.correct).length;
  document.getElementById('exc-score').textContent = `${excCorrect}/${excRows.length}`;
  document.querySelector('#evalExcTable tbody').innerHTML = excRows.map(r => `
    <tr>
      <td class="mono">${r.record_id}</td>
      <td><span class="badge badge-exception">EXCEPTION</span></td>
      <td><span class="badge ${r.pipeline_decision === 'EXCEPTION' ? 'badge-exception' : 'badge-match'}">${r.pipeline_decision}</span></td>
      <td style="font-size:0.75rem;color:var(--text-muted)">${STAGE_LABELS[r.resolved_by] || r.resolved_by}</td>
      <td class="${r.correct ? 'check-ok' : 'check-fail'}">${r.correct ? '✓' : '✗'}</td>
    </tr>`).join('');

  // Resolvable table
  const resCorrect = resRows.filter(r => r.correct).length;
  document.getElementById('res-score').textContent = `${resCorrect}/${resRows.length}`;
  document.querySelector('#evalResTable tbody').innerHTML = resRows.map(r => `
    <tr>
      <td class="mono">${r.record_id}</td>
      <td><span class="badge badge-borderline">${r.challenge_type}</span></td>
      <td><span class="badge badge-match">MATCH</span></td>
      <td><span class="badge ${r.pipeline_decision === 'MATCH' ? 'badge-match' : 'badge-exception'}">${r.pipeline_decision}</span></td>
      <td style="font-size:0.75rem;color:var(--text-muted)">${STAGE_LABELS[r.resolved_by] || r.resolved_by}</td>
      <td class="${r.correct ? 'check-ok' : 'check-fail'}">${r.correct ? '✓' : '✗'}</td>
    </tr>`).join('');
}
