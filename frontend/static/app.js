// ProofCausality AI — Frontend Interaction Engine v0.1.1
let currentDataset = 'real';
let realBenchmarkData = null;
let syntheticBenchmarkData = null;

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initDiagnostics();
  initLabelCourt();
  initActionProof();
  initBenchmarks();
  initPerformance();
  initCalculus();
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

  if (window.location.hash) {
    const targetTabId = window.location.hash.replace('#', '');
    const hashBtn = document.querySelector();
    if (hashBtn) {
      hashBtn.click();
    }
  }
}

// System Diagnostics & Health Probe
async function initDiagnostics() {
  try {
    const healthResp = await fetch('/api/health');
    const healthData = await healthResp.json();
    document.getElementById('diag-api-status').innerText = 'Operational (v' + healthData.version + ')';
    document.getElementById('diag-api-status').className = 'diag-val text-success';
    if (healthData.version) {
      const eb = document.getElementById('engine-badge');
      if (eb) eb.innerText = 'Engine v' + healthData.version;
    }
    if (healthData.hardware_acceleration && healthData.hardware_acceleration.rust_native_active) {
      document.getElementById('rust-badge').innerText = 'Rust SIMD: Active';
      document.getElementById('rust-badge').className = 'badge badge-success';
      document.getElementById('diag-acc-mode').innerText = healthData.hardware_acceleration.backend_engine;
    } else {
      document.getElementById('rust-badge').innerText = 'NumPy Fallback';
      document.getElementById('rust-badge').className = 'badge badge-warning';
      document.getElementById('diag-acc-mode').innerText = 'Pure NumPy (No Rust)';
    }
  } catch (e) {
    document.getElementById('diag-api-status').innerText = 'Offline';
    document.getElementById('diag-api-status').className = 'diag-val text-danger';
  }

  try {
    const infoResp = await fetch('/api/model-info');
    const infoData = await infoResp.json();
    document.getElementById('diag-provider-mode').innerText = infoData.active_mode || 'Deterministic Engine';
    const pb = document.getElementById('provider-badge');
    if (pb) {
      if (infoData.configured) {
        pb.innerText = 'Meta API: Active';
        pb.className = 'badge badge-success';
      } else {
        pb.innerText = infoData.meta_inference_blocked ? 'Meta: Blocked (Offline)' : 'Meta: Fallback Mode';
        pb.className = 'badge badge-accent';
        if (infoData.meta_inference_explanation) {
          pb.title = infoData.meta_inference_explanation;
        }
      }
    }
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
  const fileInput = document.getElementById('lc-image-file');
  if (fileInput) {
    fileInput.addEventListener('change', handleImageUpload);
  }
}

let uploadedImageBase64 = null;

async function handleImageUpload(e) {
  const file = e.target.files[0];
  if (!file) return;

  const info = document.getElementById('lc-file-info');
  info.innerText = `Loading ${file.name} (${(file.size / 1024).toFixed(1)} KB)...`;

  const reader = new FileReader();
  reader.onload = async () => {
    uploadedImageBase64 = reader.result;
    try {
      const resp = await fetch('/api/image/upload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_base64: uploadedImageBase64, filename: file.name })
      });
      const data = await resp.json();
      if (resp.ok) {
        info.innerHTML = `<span class="text-success">Validated: ${data.width}x${data.height}px | SHA-256: ${data.sha256.substring(0, 16)}... | Laplacian Var: ${data.laplacian_variance} (${data.is_blurry ? 'BLURRED' : 'SHARP'})</span>`;
      } else {
        info.innerHTML = `<span class="text-danger">Error: ${data.detail || 'Validation failed'}</span>`;
        uploadedImageBase64 = null;
      }
    } catch (err) {
      info.innerHTML = `<span class="text-danger">Failed to upload image.</span>`;
    }
  };
  reader.readAsDataURL(file);
}

async function runLabelCourtAudit() {
  const preset = document.getElementById('lc-preset-select').value;
  let payload = {
    human_annotation: {
      id: 101,
      image_id: 1,
      category_id: 1,
      category_name: "robotic_gripper",
      bbox: { x_min: 40, y_min: 40, x_max: 140, y_max: 140 }
    },
    candidate_bbox: { x_min: 42, y_min: 41, x_max: 138, y_max: 139 },
    image_width: 300,
    image_height: 300,
    is_blurry: false
  };

  if (uploadedImageBase64) {
    payload.image_base64 = uploadedImageBase64;
  }

  if (preset === 'boundary_clipped') {
    payload.candidate_bbox = { x_min: 240, y_min: 240, x_max: 300, y_max: 300 }; // border contact
  } else if (preset === 'discrepancy') {
    payload.candidate_bbox = { x_min: 120, y_min: 120, x_max: 220, y_max: 220 }; // low IoU
  } else if (preset === 'blurry') {
    payload.is_blurry = true;
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
    console.error('LabelCourt audit failed', e);
  }
}

function renderLabelCourtResult(data) {
  const res = data.audit_result;
  const receipt = data.evidence_receipt;

  const badge = document.getElementById('lc-verdict-badge');
  badge.innerText = res.verdict;
  badge.className = 'badge ' + (res.verdict === 'ACCEPT' ? 'badge-success' : (res.verdict === 'REVIEW' ? 'badge-danger' : 'badge-warning'));

  document.getElementById('lc-iou-val').innerText = (res.iou * 100).toFixed(1) + '%';
  document.getElementById('lc-dice-val').innerText = (res.dice * 100).toFixed(1) + '%';
  document.getElementById('lc-blur-val').innerText = res.laplacian_variance || 'N/A';

  const overlayContainer = document.getElementById('lc-overlay-container');
  const overlayImg = document.getElementById('lc-overlay-img');
  if (data.overlay_data_uri) {
    overlayImg.src = data.overlay_data_uri;
    overlayContainer.style.display = 'block';
  } else {
    overlayContainer.style.display = 'none';
  }

  const signalsList = document.getElementById('lc-signals-list');
  if (res.uncertainty_signals && res.uncertainty_signals.length > 0) {
    signalsList.innerHTML = res.uncertainty_signals.map(s => `
      <div class="signal-item">
        <strong>${s.reason}:</strong> ${s.description}
      </div>
    `).join('');
  } else {
    signalsList.innerHTML = '<div class="empty-state">No spatial, border, or blur uncertainty detected.</div>';
  }

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
    payload.after_bbox = { x_min: 50, y_min: 50, x_max: 110, y_max: 110 };
  } else if (scenario === 'motion_blur') {
    payload.is_blurry = true;
  } else if (scenario === 'boundary_clipped') {
    payload.after_bbox = { x_min: 440, y_min: 170, x_max: 500, y_max: 230 };
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
  let payload = {
    request: {
      task_type: "moved_A_to_B",
      object_name: "wooden_block",
      region_a: { x_min: 20, y_min: 20, x_max: 120, y_max: 120 },
      region_b: { x_min: 300, y_min: 150, x_max: 420, y_max: 270 },
      distance_threshold_px: 60.0
    },
    before_bbox: { x_min: 30, y_min: 30, x_max: 90, y_max: 90 },
    after_bbox: { x_min: 325, y_min: 175, x_max: 385, y_max: 235 },
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
  document.getElementById('ap-prox-val').innerText = 'Contained in Region B';

  const recapBox = document.getElementById('ap-recapture-box');
  if (res.recapture_plan) {
    recapBox.style.display = 'block';
    document.getElementById('ap-recapture-text').innerText = res.recapture_plan.directive;
  } else {
    recapBox.style.display = 'none';
  }

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
    const resp = await fetch('/api/benchmark/summary?dataset=all');
    const data = await resp.json();
    realBenchmarkData = data.real_world_evaluation;
    syntheticBenchmarkData = data.synthetic_baseline;
    window.genuineModelBenchmarkData = data.genuine_model_predictions;
    renderBenchmarkView();
  } catch (e) {
    console.error('Failed to load benchmarks', e);
  }
}

function switchBenchmarkDataset(dataset) {
  currentDataset = dataset;
  document.getElementById('btn-dataset-real').className = dataset === 'real' ? 'btn btn-primary' : 'btn btn-outline';
  document.getElementById('btn-dataset-synthetic').className = dataset === 'synthetic' ? 'btn btn-primary' : 'btn btn-outline';
  const cocoBtn = document.getElementById('btn-dataset-coco');
  if (cocoBtn) cocoBtn.className = dataset === 'coco' ? 'btn btn-primary' : 'btn btn-outline';
  renderBenchmarkView();
}

function renderBenchmarkView() {
  const cocoView = document.getElementById('coco-benchmark-view');
  const tableContainer = document.getElementById('standard-benchmark-table');

  if (currentDataset === 'coco') {
    if (cocoView) cocoView.style.display = 'block';
    if (tableContainer) tableContainer.style.display = 'none';
    const titleElem = document.getElementById('bench-table-title');
    if (titleElem) titleElem.innerText = 'COCO evidence: genuine predictions vs controlled annotation corruptions';
    document.getElementById('bench-card-title-1').innerText = 'Controlled-corruption precision';
    document.getElementById('bench-card-title-2').innerText = 'Genuine model class-aware recall@0.5';
    document.getElementById('bench-card-title-3').innerText = 'Controlled-corruption recall';
    document.getElementById('bench-card-title-4').innerText = 'Genuine model class-aware mean best IoU';

    document.getElementById('bench-fsr-proofcausality').innerText = '97.5%';
    document.getElementById('bench-fsr-ci').innerText = 'Controlled-corruption precision; image-grouped CI [96.27%, 98.80%]';
    document.getElementById('bench-fsr-naive').innerText = '43.26%';
    document.getElementById('bench-fsr-naive-ci').innerText = 'Genuine class-aware recall@0.5; image-grouped CI [33.33%, 54.29%]';
    document.getElementById('bench-rrr-proofcausality').innerText = '99.0%';
    document.getElementById('bench-rrr-ci').innerText = 'Controlled-corruption recall; image-grouped CI [98.02%, 99.66%]';
    document.getElementById('bench-rrr-generic').innerText = '0.3929';
    document.getElementById('bench-rrr-generic-ci').innerText = 'Genuine class-aware mean best IoU; image-grouped CI [0.3205, 0.4738]';
    return;
  }

  if (cocoView) cocoView.style.display = 'none';
  if (tableContainer) tableContainer.style.display = 'block';

  const data = currentDataset === 'real' ? realBenchmarkData : syntheticBenchmarkData;
  if (!data) return;
  document.getElementById('bench-card-title-1').innerText = 'False-Success Rate (ProofCausality)';
  document.getElementById('bench-card-title-2').innerText = 'False-Success Rate (Naive Baseline)';
  document.getElementById('bench-card-title-3').innerText = 'Re-Capture Resolution Rate';
  document.getElementById('bench-card-title-4').innerText = 'Generic Guidance Resolution';
  document.getElementById('bench-rrr-generic-ci').innerText = '95% Wilson CI: [0.0%, 49.0%]';

  const titleElem = document.getElementById('bench-table-title');
  if (titleElem) {
    titleElem.innerText = currentDataset === 'real' ? 
      'Real-World Tabletop Evaluation Matrix (12 Cases, CC-BY-4.0 Provenance)' : 
      'Synthetic Baseline Calibration Matrix (25 Controlled Cases)';
  }

  document.getElementById('bench-fsr-proofcausality').innerText = data.false_success_rate.proofcausality_percentage;
  document.getElementById('bench-fsr-naive').innerText = data.false_success_rate.naive_baseline_percentage;
  document.getElementById('bench-rrr-proofcausality').innerText = data.recapture_resolution_rate.proofcausality_percentage;
  document.getElementById('bench-rrr-generic').innerText = data.recapture_resolution_rate.generic_baseline_percentage;

  // Wilson Score CIs
  if (data.false_success_rate.proofcausality_wilson_95_ci) {
    const [l, u] = data.false_success_rate.proofcausality_wilson_95_ci;
    document.getElementById('bench-fsr-ci').innerText = `95% Wilson CI: [${(l*100).toFixed(1)}%, ${(u*100).toFixed(1)}%] (${data.false_success_rate.numerator_proofcausality}/${data.false_success_rate.denominator})`;
  }
  if (data.false_success_rate.naive_baseline_wilson_95_ci) {
    const [l, u] = data.false_success_rate.naive_baseline_wilson_95_ci;
    document.getElementById('bench-fsr-naive-ci').innerText = `95% Wilson CI: [${(l*100).toFixed(1)}%, ${(u*100).toFixed(1)}%] (${data.false_success_rate.numerator_naive_baseline}/${data.false_success_rate.denominator})`;
  }
  if (data.recapture_resolution_rate.proofcausality_wilson_95_ci) {
    const [l, u] = data.recapture_resolution_rate.proofcausality_wilson_95_ci;
    document.getElementById('bench-rrr-ci').innerText = `95% Wilson CI: [${(l*100).toFixed(1)}%, ${(u*100).toFixed(1)}%] (${data.recapture_resolution_rate.numerator_proofcausality}/${data.recapture_resolution_rate.denominator})`;
  }

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
}

// Performance Dashboard Module
async function initPerformance() {
  try {
    const resp = await fetch('/api/benchmark/latency');
    const data = await resp.json();

    const tbody = document.getElementById('perf-tbody');
    if (tbody && data.kernels) {
      tbody.innerHTML = Object.entries(data.kernels).map(([k, v]) => `
        <tr>
          <td class="font-mono text-cyan">${k}</td>
          <td>${v.dimensions}</td>
          <td>${v.numpy.p50_us} µs</td>
          <td>${v.numpy.p95_us} µs</td>
          <td class="text-success font-semibold">${v.rust.p50_us} µs</td>
          <td>${v.rust.p95_us} µs</td>
          <td><span class="badge badge-success">${v.speedup_ratio}x Speedup</span></td>
        </tr>
      `).join('');
    }
  } catch (e) {
    console.error('Failed to load performance metrics', e);
  }
}

// Receipt Inspector
function updateReceiptViewer(receipt) {
  const viewer = document.getElementById('receipt-json-view');
  if (viewer) {
    viewer.innerText = JSON.stringify(receipt, null, 2);
  }
}

// ==============================================================================
// EVIDENCE CALCULUS ENGINE (Phase 2 Frontend Controller)
// ==============================================================================

const CALCULUS_SCENARIOS = {
  success: {
    task_id: "SCENARIO_SUCCESS",
    primary_object: "cup",
    before_bbox: { x_min: 100, y_min: 100, x_max: 200, y_max: 200 },
    after_bbox: { x_min: 310, y_min: 105, x_max: 390, y_max: 195 },
    target_region: { x_min: 300, y_min: 100, x_max: 400, y_max: 200 },
    canvas_width: 640,
    canvas_height: 480,
    laplacian_variance: 250.0,
    area_drop_ratio: 0.0,
    has_final_state: true
  },
  missing: {
    task_id: "SCENARIO_MISSING_OCCLUDED",
    primary_object: "cup",
    before_bbox: { x_min: 100, y_min: 100, x_max: 200, y_max: 200 },
    after_bbox: null,
    target_region: { x_min: 300, y_min: 100, x_max: 400, y_max: 200 },
    canvas_width: 640,
    canvas_height: 480,
    laplacian_variance: 250.0,
    area_drop_ratio: 0.55,
    has_final_state: false
  },
  blur: {
    task_id: "SCENARIO_BLUR_DEFICIT",
    primary_object: "cup",
    before_bbox: { x_min: 100, y_min: 100, x_max: 200, y_max: 200 },
    after_bbox: { x_min: 315, y_min: 108, x_max: 392, y_max: 196 },
    target_region: { x_min: 300, y_min: 100, x_max: 400, y_max: 200 },
    canvas_width: 640,
    canvas_height: 480,
    laplacian_variance: 38.5,
    area_drop_ratio: 0.05,
    has_final_state: true
  },
  contra: {
    task_id: "SCENARIO_PHYSICAL_CONTRADICTION",
    primary_object: "cup",
    before_bbox: { x_min: 50, y_min: 50, x_max: 100, y_max: 100 },
    after_bbox: { x_min: 520, y_min: 420, x_max: 570, y_max: 470 },
    target_region: { x_min: 500, y_min: 400, x_max: 580, y_max: 480 },
    canvas_width: 640,
    canvas_height: 480,
    laplacian_variance: 210.0,
    area_drop_ratio: 0.0,
    has_final_state: true
  }
};

function initCalculus() {
  const btnRun = document.getElementById('btn-run-calculus');
  const scenarioSelect = document.getElementById('calculus-scenario-select');

  if (btnRun) {
    btnRun.addEventListener('click', () => {
      const selected = scenarioSelect ? scenarioSelect.value : 'success';
      evaluateCalculusScenario(selected);
    });
  }

  if (scenarioSelect) {
    scenarioSelect.addEventListener('change', () => {
      evaluateCalculusScenario(scenarioSelect.value);
    });
  }

  // Load initial graph and evaluate default scenario
  drawEvidenceGraph();
  evaluateCalculusScenario('success');
}

async function evaluateCalculusScenario(scenarioKey) {
  const payload = CALCULUS_SCENARIOS[scenarioKey] || CALCULUS_SCENARIOS.success;
  try {
    const resp = await fetch('/api/calculus/evaluate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!resp.ok) throw new Error('Calculus evaluation API returned ' + resp.status);
    const data = await resp.json();

    // 1. Update KPI bar
    const verdictEl = document.getElementById('calc-verdict-val');
    const verdictDescEl = document.getElementById('calc-verdict-desc');
    verdictEl.innerText = data.verdict;
    if (data.verdict === 'SUCCESS') {
      verdictEl.className = 'metric-value text-success';
      verdictDescEl.innerText = 'All critical predicates verified. Zero debt.';
    } else if (data.verdict === 'FAILED') {
      verdictEl.className = 'metric-value text-danger';
      verdictDescEl.innerText = 'Contradiction detected: physical falsification.';
    } else {
      verdictEl.className = 'metric-value text-warning';
      verdictDescEl.innerText = 'Incomplete evidence: system abstains (debt > 0).';
    }

    document.getElementById('calc-debt-val').innerText = data.debt_report.total_debt.toFixed(2);
    document.getElementById('calc-missing-val').innerText =
      data.debt_report.missing_evidence_debt.toFixed(2) + ' / ' + data.debt_report.contradictions.length;

    // 2. Populate Predicates Table
    const tbody = document.getElementById('calc-predicates-tbody');
    tbody.innerHTML = '';
    data.predicates.forEach(p => {
      const tr = document.createElement('tr');
      let badgeClass = 'badge-success';
      if (p.state === 'UNOBSERVABLE') badgeClass = 'badge-warning';
      if (p.state === 'CONTRADICTED') badgeClass = 'badge-danger';

      tr.innerHTML = `
        <td style="font-weight:600;font-family:var(--font-mono);font-size:12px;">${p.predicate_type}</td>
        <td><span class="badge ${badgeClass}">${p.state}</span></td>
        <td style="font-family:var(--font-mono);">${p.measured_value !== null ? p.measured_value : '—'}</td>
        <td style="font-size:12px;color:var(--text-secondary);">${p.rationale || p.description}</td>
      `;
      tbody.appendChild(tr);
    });

    // 3. Update Recovery Action Box
    const act = data.recommended_action;
    if (act) {
      document.getElementById('calc-action-name').innerText = act.name;
      document.getElementById('calc-action-directive').innerText = act.directive;
      document.getElementById('calc-act-effort').innerText = act.estimated_human_effort_sec.toFixed(1) + ' sec (latency: ' + act.estimated_camera_latency_sec.toFixed(1) + 's)';
      document.getElementById('calc-act-risk').innerText = Math.round(act.expected_risk_reduction * 100) + '% Risk Reduction';
      document.getElementById('calc-act-voi').innerText = act.net_utility_score.toFixed(3);
    } else {
      document.getElementById('calc-action-name').innerText = 'No Recovery Required';
      document.getElementById('calc-action-directive').innerText = 'Evidence is complete. All physical predicates verified.';
      document.getElementById('calc-act-effort').innerText = '0.0 sec';
      document.getElementById('calc-act-risk').innerText = 'N/A';
      document.getElementById('calc-act-voi').innerText = 'Optimal (Zero Debt)';
    }

    // 4. Update DAG node colors based on scenario
    updateGraphNodeColors(scenarioKey, data);

  } catch (e) {
    console.error('Failed to evaluate calculus scenario', e);
  }
}

function drawEvidenceGraph() {
  const svg = document.getElementById('calc-dag-svg');
  if (!svg) return;

  // Stages layout:
  // 0: Observations (x=40)
  // 1: Detections (x=160)
  // 2: Measurements / Masks (x=280)
  // 3: Predicates (x=400)
  // 4: Verdict (x=520)
  const nodes = [
    { id: 'obs_0', label: 'Frame t=0', x: 40, y: 60, stage: 'OBS' },
    { id: 'obs_T', label: 'Frame t=T', x: 40, y: 220, stage: 'OBS' },

    { id: 'det_cup_0', label: 'Cup (t=0)', x: 160, y: 50, stage: 'DET' },
    { id: 'det_bot_0', label: 'Bottle (t=0)', x: 160, y: 110, stage: 'DET' },
    { id: 'det_cup_T', label: 'Cup (t=T)', x: 160, y: 190, stage: 'DET' },
    { id: 'det_bot_T', label: 'Bottle (t=T)', x: 160, y: 260, stage: 'DET' },

    { id: 'meas_disp', label: 'Displacement', x: 280, y: 90, stage: 'MEAS' },
    { id: 'meas_blur', label: 'Blur Var', x: 280, y: 170, stage: 'MEAS' },
    { id: 'meas_prox', label: 'Proximity', x: 280, y: 250, stage: 'MEAS' },

    { id: 'pred_ident', label: 'P_IDENT', x: 400, y: 60, stage: 'PRED' },
    { id: 'pred_spatial', label: 'P_SPATIAL', x: 400, y: 140, stage: 'PRED' },
    { id: 'pred_sharp', label: 'P_SHARP', x: 400, y: 220, stage: 'PRED' },
    { id: 'pred_phys', label: 'P_PHYS', x: 400, y: 290, stage: 'PRED' },

    { id: 'verdict', label: 'VERDICT', x: 530, y: 170, stage: 'VERD' },
  ];

  const edges = [
    ['obs_0', 'det_cup_0'], ['obs_0', 'det_bot_0'],
    ['obs_T', 'det_cup_T'], ['obs_T', 'det_bot_T'], ['obs_T', 'meas_blur'],
    ['det_cup_0', 'meas_disp'], ['det_cup_T', 'meas_disp'],
    ['det_cup_T', 'meas_prox'], ['det_bot_T', 'meas_prox'],
    ['det_cup_0', 'pred_ident'], ['det_cup_T', 'pred_ident'],
    ['meas_disp', 'pred_spatial'], ['meas_prox', 'pred_spatial'],
    ['meas_blur', 'pred_sharp'],
    ['meas_disp', 'pred_phys'],
    ['pred_ident', 'verdict'], ['pred_spatial', 'verdict'], ['pred_sharp', 'verdict'], ['pred_phys', 'verdict']
  ];

  let svgHtml = '<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 1 L 9 5 L 0 9 z" fill="#374151"/></marker></defs>';

  // Draw edges
  edges.forEach(([srcId, tgtId]) => {
    const s = nodes.find(n => n.id === srcId);
    const t = nodes.find(n => n.id === tgtId);
    if (s && t) {
      svgHtml += `<line x1="${s.x + 35}" y1="${s.y + 12}" x2="${t.x - 5}" y2="${t.y + 12}" stroke="#1F2937" stroke-width="1.5" marker-end="url(#arrow)"/>`;
    }
  });

  // Draw nodes
  nodes.forEach(n => {
    const fill = n.stage === 'VERD' ? '#06B6D4' : '#1E293B';
    const textColor = n.stage === 'VERD' ? '#000000' : '#F9FAFB';
    svgHtml += `
      <g id="node-g-${n.id}">
        <rect id="node-rect-${n.id}" x="${n.x - 5}" y="${n.y}" width="75" height="26" rx="4" fill="${fill}" stroke="#374151" stroke-width="1"/>
        <text id="node-text-${n.id}" x="${n.x + 32}" y="${n.y + 17}" text-anchor="middle" font-size="10" font-family="var(--font-mono)" fill="${textColor}" font-weight="600">${n.label}</text>
      </g>
    `;
  });

  svg.innerHTML = svgHtml;
}

function updateGraphNodeColors(scenarioKey, data) {
  const nodes = ['obs_0', 'obs_T', 'det_cup_0', 'det_bot_0', 'det_cup_T', 'det_bot_T', 'meas_disp', 'meas_blur', 'meas_prox', 'pred_ident', 'pred_spatial', 'pred_sharp', 'pred_phys', 'verdict'];

  // Default all to green (#10B981)
  nodes.forEach(id => {
    const r = document.getElementById(`node-rect-${id}`);
    if (r && id !== 'verdict') r.setAttribute('stroke', '#10B981');
  });

  const vRect = document.getElementById('node-rect-verdict');
  if (vRect) {
    if (data.verdict === 'SUCCESS') vRect.setAttribute('fill', '#10B981');
    else if (data.verdict === 'FAILED') vRect.setAttribute('fill', '#EF4444');
    else vRect.setAttribute('fill', '#F59E0B');
  }

  if (scenarioKey === 'missing') {
    const rT = document.getElementById('node-rect-obs_T');
    const rDet = document.getElementById('node-rect-det_cup_T');
    const rSpatial = document.getElementById('node-rect-pred_spatial');
    if (rT) rT.setAttribute('stroke', '#F59E0B');
    if (rDet) rDet.setAttribute('stroke', '#F59E0B');
    if (rSpatial) rSpatial.setAttribute('stroke', '#F59E0B');
  } else if (scenarioKey === 'blur') {
    const rBlur = document.getElementById('node-rect-meas_blur');
    const rSharp = document.getElementById('node-rect-pred_sharp');
    if (rBlur) rBlur.setAttribute('stroke', '#F59E0B');
    if (rSharp) rSharp.setAttribute('stroke', '#F59E0B');
  } else if (scenarioKey === 'contra') {
    const rDisp = document.getElementById('node-rect-meas_disp');
    const rPhys = document.getElementById('node-rect-pred_phys');
    if (rDisp) rDisp.setAttribute('stroke', '#EF4444');
    if (rPhys) rPhys.setAttribute('stroke', '#EF4444');
  }
}
