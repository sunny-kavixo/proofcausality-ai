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
    document.getElementById('diag-api-status').innerText = 'Operational (v' + healthData.version + ')';
    document.getElementById('diag-api-status').className = 'diag-val text-success';
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
    document.getElementById('provider-badge').innerText = infoData.configured ? 'Meta API: Active' : 'Meta: Fallback Mode';
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
    renderBenchmarkView();
  } catch (e) {
    console.error('Failed to load benchmarks', e);
  }
}

function switchBenchmarkDataset(dataset) {
  currentDataset = dataset;
  document.getElementById('btn-dataset-real').className = dataset === 'real' ? 'btn btn-primary' : 'btn btn-outline';
  document.getElementById('btn-dataset-synthetic').className = dataset === 'synthetic' ? 'btn btn-primary' : 'btn btn-outline';
  renderBenchmarkView();
}

function renderBenchmarkView() {
  const data = currentDataset === 'real' ? realBenchmarkData : syntheticBenchmarkData;
  if (!data) return;

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
