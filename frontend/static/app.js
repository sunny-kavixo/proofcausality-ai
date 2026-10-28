// ProofCausality AI — Frontend Interaction Engine
document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initDiagnostics();
  initLabelCourt();
  initActionProof();
  initBenchmarks();
});

// Tab Switching
function initTabs() {
  const tabs = document.querySelectorAll('.nav-tab');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
      tab.classList.add('active');
      const target = document.getElementById(tab.dataset.tab);
      if (target) target.classList.add('active');
    });
  });
}

// System Diagnostics & Health Probe
async function initDiagnostics() {
  try {
    const healthResp = await fetch('/api/health');
    const healthData = await healthResp.json();
    document.getElementById('diag-api-status').innerText = 'Operational (HTTP 200)';
    document.getElementById('diag-api-status').className = 'status-val text-success';
  } catch (e) {
    document.getElementById('diag-api-status').innerText = 'Offline';
    document.getElementById('diag-api-status').className = 'status-val text-danger';
  }

  try {
    const infoResp = await fetch('/api/model-info');
    const infoData = await infoResp.json();
    document.getElementById('diag-provider-mode').innerText = infoData.active_mode || 'Deterministic Engine';
    document.getElementById('provider-badge').innerText = infoData.configured ? 'Meta API: Active' : 'Meta: Fallback Engine';
  } catch (e) {
    console.error('Failed to load model diagnostics', e);
  }
}

// LabelCourt Module
function initLabelCourt() {
  const btnAudit = document.getElementById('btn-run-labelcourt');
  if (btnAudit) {
    btnAudit.addEventListener('click', runLabelCourtAudit);
  }
}

async function runLabelCourtAudit() {
  const preset = document.getElementById('lc-preset-select').value;
  let payload = {
    human_annotation: {
      id: 101,
      image_id: 1,
      category_id: 1,
      category_name: "coffee_cup",
      bbox: { x_min: 50, y_min: 50, x_max: 200, y_max: 200 }
    },
    candidate_bbox: { x_min: 50, y_min: 50, x_max: 200, y_max: 200 },
    image_width: 400,
    image_height: 300,
    is_blurry: false,
    class_candidate: "coffee_cup"
  };

  if (preset === 'shifted') {
    payload.candidate_bbox = { x_min: 90, y_min: 90, x_max: 240, y_max: 240 };
  } else if (preset === 'blurry') {
    payload.is_blurry = true;
  } else if (preset === 'mislabeled') {
    payload.class_candidate = "water_bottle";
  }

  try {
    const resp = await fetch('/api/labelcourt/audit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await resp.json();
    renderLabelCourtResult(data);
  } catch (e) {
    console.error('Audit failed', e);
  }
}

function renderLabelCourtResult(data) {
  const res = data.audit_result;
  const receipt = data.evidence_receipt;

  // Verdict Badge
  const badge = document.getElementById('lc-verdict-badge');
  badge.innerText = res.verdict;
  badge.className = 'badge ' + (res.verdict === 'ACCEPT' ? 'badge-success' : (res.verdict === 'REVIEW' ? 'badge-warning' : 'badge-danger'));

  // Metrics
  document.getElementById('lc-iou-val').innerText = (res.iou * 100).toFixed(1) + '%';
  document.getElementById('lc-boundary-val').innerText = (res.boundary_agreement * 100).toFixed(1) + '%';
  document.getElementById('lc-disp-val').innerText = (res.bbox_displacement * 100).toFixed(1) + '%';
  document.getElementById('lc-class-val').innerText = res.class_match ? 'Match' : 'Mismatch';
  document.getElementById('lc-class-val').className = 'metric-val ' + (res.class_match ? 'text-success' : 'text-danger');

  // Heatmap Overlay
  const overlayView = document.getElementById('lc-overlay-view');
  if (res.disagreement_heatmap_base64) {
    overlayView.innerHTML = `<img src="data:image/png;base64,${res.disagreement_heatmap_base64}" alt="Disagreement Heatmap">`;
  }

  // Signals
  const signalsList = document.getElementById('lc-signals-list');
  if (res.uncertainty_signals && res.uncertainty_signals.length > 0) {
    signalsList.innerHTML = res.uncertainty_signals.map(s => `
      <div class="signal-item">
        <strong>${s.reason}:</strong> ${s.description}
      </div>
    `).join('');
  } else {
    signalsList.innerHTML = '<div class="empty-state">No spatial or blur uncertainty detected.</div>';
  }

  // Receipt ID & Quick Viewer
  document.getElementById('lc-receipt-id').innerText = 'Receipt: ' + receipt.report_id;
  updateReceiptViewer(receipt);
}

// ActionProof Module
function initActionProof() {
  const btnVerify = document.getElementById('btn-run-actionproof');
  if (btnVerify) {
    btnVerify.addEventListener('click', runActionProof);
  }
  const btnTrigger = document.getElementById('btn-trigger-guided-recapture');
  if (btnTrigger) {
    btnTrigger.addEventListener('click', runActionProofGuidedRecapture);
  }
}

async function runActionProof() {
  const scenario = document.getElementById('ap-scenario-select').value;
  let payload = {
    request: {
      task_type: "moved_A_to_B",
      object_name: "wooden_block",
      region_a: { x_min: 20, y_min: 20, x_max: 120, y_max: 120 },
      region_b: { x_min: 300, y_min: 150, x_max: 420, y_max: 270 },
      distance_threshold_px: 60.0
    },
    before_bbox: { x_min: 30, y_min: 30, x_max: 90, y_max: 90 },
    after_bbox: { x_min: 320, y_min: 170, x_max: 380, y_max: 230 },
    image_width: 500,
    image_height: 300,
    is_blurry: false,
    area_drop: false,
    touches_border: false,
    has_final_state: true
  };

  if (scenario === 'incomplete_move') {
    payload.after_bbox = { x_min: 50, y_min: 50, x_max: 110, y_max: 110 }; // stays in A
  } else if (scenario === 'motion_blur') {
    payload.is_blurry = true;
  } else if (scenario === 'boundary_clipped') {
    payload.after_bbox = { x_min: 440, y_min: 170, x_max: 500, y_max: 230 }; // touches border
  } else if (scenario === 'hand_occlusion') {
    payload.area_drop = true;
  }

  try {
    const resp = await fetch('/api/actionproof/verify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await resp.json();
    renderActionProofResult(data);
  } catch (e) {
    console.error('ActionProof verification failed', e);
  }
}

async function runActionProofGuidedRecapture() {
  // Re-run with clean resolved coordinates
  let payload = {
    request: {
      task_type: "moved_A_to_B",
      object_name: "wooden_block",
      region_a: { x_min: 20, y_min: 20, x_max: 120, y_max: 120 },
      region_b: { x_min: 300, y_min: 150, x_max: 420, y_max: 270 },
      distance_threshold_px: 60.0
    },
    before_bbox: { x_min: 30, y_min: 30, x_max: 90, y_max: 90 },
    after_bbox: { x_min: 325, y_min: 175, x_max: 385, y_max: 235 }, // clean placement
    image_width: 500,
    image_height: 300,
    is_blurry: false,
    area_drop: false,
    touches_border: false,
    has_final_state: true
  };

  const resp = await fetch('/api/actionproof/verify', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  const data = await resp.json();
  renderActionProofResult(data);
}

function renderActionProofResult(data) {
  const res = data.verification_result;
  const receipt = data.evidence_receipt;

  const badge = document.getElementById('ap-verdict-badge');
  badge.innerText = res.verdict;
  badge.className = 'badge ' + (res.verdict === 'SUCCESS' ? 'badge-success' : (res.verdict === 'FAILED' ? 'badge-danger' : 'badge-warning'));

  document.getElementById('ap-disp-val').innerText = res.displacement_px + ' px';
  document.getElementById('ap-conf-val').innerText = (res.confidence * 100).toFixed(0) + '%';

  // ReCapture Box
  const recapBox = document.getElementById('ap-recapture-box');
  if (res.recapture_plan) {
    recapBox.style.display = 'block';
    document.getElementById('ap-recapture-text').innerText = res.recapture_plan.directive;
  } else {
    recapBox.style.display = 'none';
  }

  // Signals
  const signalsList = document.getElementById('ap-signals-list');
  if (res.uncertainty_signals && res.uncertainty_signals.length > 0) {
    signalsList.innerHTML = res.uncertainty_signals.map(s => `
      <div class="signal-item">
        <strong>${s.reason} (${s.severity}):</strong> ${s.description}
      </div>
    `).join('');
  } else {
    signalsList.innerHTML = '<div class="empty-state">Physical action satisfies all visibility and displacement criteria.</div>';
  }

  document.getElementById('ap-receipt-id').innerText = 'Receipt: ' + receipt.report_id;
  updateReceiptViewer(receipt);
}

// ReCapture Simulator
function simulateRecovery(defect) {
  const box = document.getElementById('sim-results-box');
  box.style.display = 'block';

  const directives = {
    'BLUR': {
      text: 'Directive: Hold the camera steady and allow the object to become sharp before ending the capture.',
      note: 'Camera stabilized. Laplacian variance increased from 34.2 to 248.5. Decisive SUCCESS confirmed.'
    },
    'OUT_OF_FRAME': {
      text: 'Directive: Move the camera back so the full object remains inside the frame.',
      note: 'Camera moved back 20 cm. Zero boundary contact pixels. Full containment in Region B confirmed.'
    },
    'OCCLUSION': {
      text: 'Directive: Keep the object visible after releasing it and record the final state.',
      note: 'Hand withdrawn from frame. Surface area retention restored to 98.4%. Decisive SUCCESS confirmed.'
    }
  };

  document.getElementById('sim-initial-verdict').innerText = 'UNCERTAIN';
  document.getElementById('sim-initial-verdict').className = 'badge badge-warning';
  document.getElementById('sim-directive-text').innerText = directives[defect].text;
  document.getElementById('sim-resolved-note').innerText = directives[defect].note;
}

// Benchmarks Module
async function initBenchmarks() {
  const btnRefresh = document.getElementById('btn-refresh-benchmarks');
  if (btnRefresh) {
    btnRefresh.addEventListener('click', loadBenchmarks);
  }
  loadBenchmarks();
}

async function loadBenchmarks() {
  try {
    const resp = await fetch('/api/benchmark/summary');
    const data = await resp.json();

    document.getElementById('bench-fsr-proofcausality').innerText = data.false_success_rate.proofcausality_percentage;
    document.getElementById('bench-fsr-naive').innerText = data.false_success_rate.naive_baseline_percentage;
    document.getElementById('bench-rrr-proofcausality').innerText = data.recapture_resolution_rate.proofcausality_percentage;
    document.getElementById('bench-rrr-generic').innerText = data.recapture_resolution_rate.generic_baseline_percentage;

    const tbody = document.getElementById('bench-tbody');
    tbody.innerHTML = data.detailed_records.map(r => `
      <tr>
        <td class="font-mono text-cyan">${r.case_id}</td>
        <td>${r.name}</td>
        <td><span class="badge ${r.ground_truth === 'SUCCESS' ? 'badge-success' : (r.ground_truth === 'FAILED' ? 'badge-danger' : 'badge-warning')}">${r.ground_truth}</span></td>
        <td><span class="badge ${r.proofcausality_verdict === 'SUCCESS' ? 'badge-success' : (r.proofcausality_verdict === 'FAILED' ? 'badge-danger' : 'badge-warning')}">${r.proofcausality_verdict}</span></td>
        <td><span class="badge ${r.naive_baseline_verdict === 'SUCCESS' ? 'badge-success' : 'badge-danger'}">${r.naive_baseline_verdict}</span></td>
        <td class="text-sm">${r.signals_detected.length ? r.signals_detected.join(', ') : 'None'}</td>
        <td class="text-sm">${r.recapture_resolved === true ? '<span class="text-success">✓ Resolved</span>' : (r.recapture_resolved === false ? '<span class="text-danger">Unresolved</span>' : 'N/A')}</td>
      </tr>
    `).join('');
  } catch (e) {
    console.error('Failed to load benchmarks', e);
  }
}

// Receipt Inspector
function updateReceiptViewer(receipt) {
  const viewer = document.getElementById('receipt-json-view');
  if (viewer) {
    viewer.innerText = JSON.stringify(receipt, null, 2);
  }
}
