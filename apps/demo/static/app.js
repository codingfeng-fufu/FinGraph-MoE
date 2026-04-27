/* legacy workflow demo retained for reference
const steps = ["query", "retrieve", "route", "execute", "result"];
const stepDescriptions = {
  query: "User query accepted.",
  retrieve: "Top-k similar training queries retrieved from the router index.",
  route: "Similarity-weighted vote selected one expert.",
  execute: "Chosen expert checkpoint executed on the matched graph example.",
  result: "Prediction and error computed.",
};

const queryEl = document.getElementById("query");
const fileEl = document.getElementById("document-file");
const uploadBtn = document.getElementById("upload-btn");
const docResetBtn = document.getElementById("doc-reset-btn");
const docNameEl = document.getElementById("doc-name");
const docMetaEl = document.getElementById("doc-meta");
const docPreviewEl = document.getElementById("doc-preview");
const runBtn = document.getElementById("run-btn");
const resetBtn = document.getElementById("reset-btn");
const examplesEl = document.getElementById("examples");
const detailsEl = document.getElementById("details");
const timelineEl = document.getElementById("timeline");
const hoverCardEl = document.getElementById("hover-card");
const hoverCardExpertEl = document.getElementById("hover-card-expert");
const hoverCardScoreEl = document.getElementById("hover-card-score");
const hoverCardQueryEl = document.getElementById("hover-card-query");
const statusStageEl = document.getElementById("status-stage");
const statusExpertEl = document.getElementById("status-expert");
const statusConfidenceEl = document.getElementById("status-confidence");
const statusNoteEl = document.getElementById("status-note");
const detailTabsEl = document.getElementById("detail-tabs");
const branchLayerEl = document.getElementById("retrieve-branch-layer");
const branchLinksEl = document.getElementById("retrieve-branch-links");
const branchNodesEl = document.getElementById("retrieve-branch-nodes");
const routeVoteLayerEl = document.getElementById("route-vote-layer");
const expertBadgeLayerEl = document.getElementById("expert-badge-layer");
const expertBadgeLabelEl = document.getElementById("expert-badge-label");
const executeBranchLayerEl = document.getElementById("execute-branch-layer");
const executeBranchLinksEl = document.getElementById("execute-branch-links");
const executeBranchNodesEl = document.getElementById("execute-branch-nodes");
const resultRingLayerEl = document.getElementById("result-ring-layer");
const resultRingFgEl = document.getElementById("result-ring-fg");
const resultRingValueEl = document.getElementById("result-ring-value");
let currentPayload = null;
let activeDetailStep = "query";
let currentSession = null;

const LAYOUT = {
  query: { x: 120, y: 98 },
  retrieve: { x: 340, y: 98 },
  route: { x: 560, y: 98 },
  execute: { x: 780, y: 98 },
  result: { x: 1000, y: 98 },
  retrieveBranchY: 314,
  executeBranchY: 306,
};

function setDocumentFlowState(activeStep) {
  const docSteps = ["upload", "parse", "extract", "graph"];
  document.querySelectorAll(".doc-step").forEach((node) => {
    const state = node.dataset.docStep;
    node.classList.toggle("active", state === activeStep);
    node.classList.toggle("complete", docSteps.indexOf(state) < docSteps.indexOf(activeStep));
  });
  document.querySelectorAll(".doc-flow-arrow").forEach((node, index) => {
    const activeIndex = docSteps.indexOf(activeStep);
    node.classList.toggle("active", index + 1 === activeIndex);
    node.classList.toggle("complete", index + 1 < activeIndex);
  });
}

function shortText(text, limit = 12) {
  if (text.length <= limit) return text;
  return `${text.slice(0, limit - 1)}…`;
}

function hideHoverCard() {
  hoverCardEl.classList.remove("visible");
  hoverCardEl.setAttribute("aria-hidden", "true");
}

function showHoverCard(event, neighbor) {
  const shellRect = document.querySelector(".graph-shell").getBoundingClientRect();
  const rect = event.currentTarget.getBoundingClientRect();
  hoverCardExpertEl.textContent = neighbor.expert.toUpperCase();
  hoverCardScoreEl.textContent = `Similarity ${neighbor.score.toFixed(3)}`;
  hoverCardQueryEl.textContent = neighbor.text;

  const left = rect.left - shellRect.left + rect.width / 2 - 120;
  const top = rect.top - shellRect.top - 98;
  hoverCardEl.style.left = `${Math.max(12, left)}px`;
  hoverCardEl.style.top = `${Math.max(12, top)}px`;
  hoverCardEl.classList.add("visible");
  hoverCardEl.setAttribute("aria-hidden", "false");
}

function renderRetrieveBranch(neighbors, selectedExpert) {
  branchLinksEl.innerHTML = "";
  branchNodesEl.innerHTML = "";

  const top = neighbors.slice(0, 4);
  if (!top.length) {
    branchLayerEl.classList.remove("visible", "active", "complete");
    branchLayerEl.dataset.ready = "false";
    return;
  }

  const layouts = {
    1: [340],
    2: [240, 440],
    3: [160, 340, 520],
    4: [120, 270, 410, 560],
  };
  const xs = layouts[top.length] || layouts[4];
  const y = LAYOUT.retrieveBranchY;

  top.forEach((neighbor, index) => {
    const x = xs[index];
    const link = document.createElementNS("http://www.w3.org/2000/svg", "g");
    link.setAttribute("class", "branch-link");
    if (neighbor.expert === selectedExpert) {
      link.classList.add("complete");
    }

    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", String(LAYOUT.retrieve.x));
    line.setAttribute("y1", String(LAYOUT.retrieve.y + 78));
    line.setAttribute("x2", String(x));
    line.setAttribute("y2", String(y - 24));
    link.appendChild(line);

    const arrow = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    arrow.setAttribute("class", "link-arrow");
    arrow.setAttribute(
      "points",
      `${x},${y - 24} ${x - 8},${y - 38} ${x + 8},${y - 38}`
    );
    link.appendChild(arrow);
    branchLinksEl.appendChild(link);

    const node = document.createElementNS("http://www.w3.org/2000/svg", "g");
    node.setAttribute("class", "branch-node");
    node.setAttribute("transform", `translate(${x} ${y})`);
    if (neighbor.expert === selectedExpert) {
      node.classList.add("selected");
    }
    node.addEventListener("mouseenter", (event) => showHoverCard(event, neighbor));
    node.addEventListener("mouseleave", hideHoverCard);

    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("class", "branch-circle");
    circle.setAttribute("r", "22");
    node.appendChild(circle);

    const rank = document.createElementNS("http://www.w3.org/2000/svg", "text");
    rank.setAttribute("class", "branch-rank");
    rank.setAttribute("text-anchor", "middle");
    rank.setAttribute("dy", "4");
    rank.textContent = String(index + 1);
    node.appendChild(rank);

    const expert = document.createElementNS("http://www.w3.org/2000/svg", "text");
    expert.setAttribute("class", "branch-expert");
    expert.setAttribute("text-anchor", "middle");
    expert.setAttribute("y", "38");
    expert.textContent = neighbor.expert.toUpperCase();
    node.appendChild(expert);

    const score = document.createElementNS("http://www.w3.org/2000/svg", "text");
    score.setAttribute("class", "branch-score");
    score.setAttribute("text-anchor", "middle");
    score.setAttribute("y", "54");
    score.textContent = neighbor.score.toFixed(2);
    node.appendChild(score);

    const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
    title.textContent = neighbor.text;
    node.appendChild(title);

    branchNodesEl.appendChild(node);
  });

  branchLayerEl.classList.add("visible");
  branchLayerEl.dataset.ready = "true";
}

function renderRouteVotes(route) {
  routeVoteLayerEl.innerHTML = "";
  const scores = route.expert_scores || {};
  const expertOrder = ["sum", "count", "predict"];
  const entries = expertOrder.map((expert) => [expert, scores[expert] || 0]);
  const visibleEntries = entries.filter(([, value]) => value > 0);
  if (!visibleEntries.length) {
    routeVoteLayerEl.classList.remove("visible");
    routeVoteLayerEl.dataset.ready = "false";
    return;
  }

  const maxScore = Math.max(...entries.map(([, value]) => value), 1e-6);
  const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
  group.setAttribute("transform", "translate(600 8)");

  const card = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  card.setAttribute("class", "vote-card");
  card.setAttribute("rx", "12");
  card.setAttribute("ry", "12");
  card.setAttribute("width", "150");
  card.setAttribute("height", "78");
  group.appendChild(card);

  const title = document.createElementNS("http://www.w3.org/2000/svg", "text");
  title.setAttribute("class", "vote-title");
  title.setAttribute("x", "12");
  title.setAttribute("y", "15");
  title.textContent = "Expert vote";
  group.appendChild(title);

  entries.forEach(([expert, score], index) => {
    const row = document.createElementNS("http://www.w3.org/2000/svg", "g");
    row.setAttribute("class", "vote-row");
    if (expert === route.selected_expert) {
      row.classList.add("selected");
    }
    row.setAttribute("transform", `translate(10 ${24 + index * 17})`);

    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("class", "vote-label");
    label.setAttribute("x", "0");
    label.setAttribute("y", "10");
    label.textContent = expert.toUpperCase();
    row.appendChild(label);

    const bg = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    bg.setAttribute("class", "vote-bg");
    bg.setAttribute("x", "46");
    bg.setAttribute("y", "1");
    bg.setAttribute("width", "82");
    bg.setAttribute("height", "10");
    bg.setAttribute("rx", "2");
    bg.setAttribute("ry", "2");
    row.appendChild(bg);

    const fill = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    fill.setAttribute("class", "vote-fill");
    fill.setAttribute("x", "46");
    fill.setAttribute("y", "1");
    fill.setAttribute("width", String((score / maxScore) * 82));
    fill.setAttribute("height", "10");
    fill.setAttribute("rx", "2");
    fill.setAttribute("ry", "2");
    row.appendChild(fill);

    const scoreText = document.createElementNS("http://www.w3.org/2000/svg", "text");
    scoreText.setAttribute("class", "vote-label");
    scoreText.setAttribute("x", "136");
    scoreText.setAttribute("y", "10");
    scoreText.setAttribute("text-anchor", "end");
    scoreText.textContent = score > 0 ? score.toFixed(2) : "0";
    row.appendChild(scoreText);

    group.appendChild(row);
  });

  routeVoteLayerEl.appendChild(group);
  routeVoteLayerEl.classList.add("visible");
  routeVoteLayerEl.dataset.ready = "true";
}

function renderExpertBadge(route) {
  expertBadgeLabelEl.textContent = route.selected_expert.toUpperCase();
  expertBadgeLayerEl.classList.add("visible");
  expertBadgeLayerEl.dataset.ready = "true";
}

function renderExecuteBranch(execute) {
  executeBranchLinksEl.innerHTML = "";
  executeBranchNodesEl.innerHTML = "";

  const items = [
    { key: "CHK", value: execute.artifact.checkpoint_path.split("/").pop(), x: 675 },
    { key: "DATA", value: execute.dataset, x: 780 },
    { key: "SAMPLE", value: execute.sample_id, x: 885 },
  ];
  const y = LAYOUT.executeBranchY;

  items.forEach((item) => {
    const link = document.createElementNS("http://www.w3.org/2000/svg", "g");
    link.setAttribute("class", "branch-link complete");

    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", String(LAYOUT.execute.x));
    line.setAttribute("y1", String(LAYOUT.execute.y + 78));
    line.setAttribute("x2", String(item.x));
    line.setAttribute("y2", String(y - 24));
    link.appendChild(line);

    const arrow = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    arrow.setAttribute("class", "link-arrow");
    arrow.setAttribute(
      "points",
      `${item.x},${y - 24} ${item.x - 8},${y - 38} ${item.x + 8},${y - 38}`
    );
    link.appendChild(arrow);
    executeBranchLinksEl.appendChild(link);

    const node = document.createElementNS("http://www.w3.org/2000/svg", "g");
    node.setAttribute("class", "branch-node selected");
    node.setAttribute("transform", `translate(${item.x} ${y})`);

    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("class", "detail-circle");
    circle.setAttribute("r", "22");
    node.appendChild(circle);

    const key = document.createElementNS("http://www.w3.org/2000/svg", "text");
    key.setAttribute("class", "detail-key");
    key.setAttribute("text-anchor", "middle");
    key.setAttribute("dy", "4");
    key.textContent = item.key;
    node.appendChild(key);

    const value = document.createElementNS("http://www.w3.org/2000/svg", "text");
    value.setAttribute("class", "detail-value");
    value.setAttribute("text-anchor", "middle");
    value.setAttribute("y", "38");
    value.textContent = shortText(item.value, 14);
    const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
    title.textContent = item.value;
    value.appendChild(title);
    node.appendChild(value);

    executeBranchNodesEl.appendChild(node);
  });

  executeBranchLayerEl.classList.add("visible");
  executeBranchLayerEl.dataset.ready = "true";
}

function renderResultRing(result) {
  const answer = Math.abs(result.answer || 0);
  const relativeError = answer > 1e-8 ? Math.abs(result.abs_error) / answer : 0;
  const clamped = Math.min(relativeError, 1);
  const percent = result.abs_error > 0 ? Math.max(clamped * 100, 8) : 0;
  resultRingFgEl.setAttribute("stroke-dasharray", `${percent} ${100 - percent}`);
  resultRingValueEl.textContent = `Δ ${(relativeError * 100).toFixed(2)}%`;

  let color = "#78a55a";
  if (relativeError > 0.3) {
    color = "#b8501f";
  } else if (relativeError > 0.12) {
    color = "#d28628";
  }
  resultRingFgEl.style.stroke = color;
  resultRingLayerEl.classList.add("visible");
  resultRingLayerEl.dataset.ready = "true";
}

function setStepState(activeStep) {
  document.querySelectorAll(".workflow-node").forEach((node) => {
    const state = node.dataset.step;
    node.classList.toggle("active", state === activeStep);
    node.classList.toggle("complete", steps.indexOf(state) < steps.indexOf(activeStep));
  });

  document.querySelectorAll(".workflow-link").forEach((edge) => {
    const [fromStep, toStep] = edge.dataset.link.split("-");
    const fromIndex = steps.indexOf(fromStep);
    const toIndex = steps.indexOf(toStep);
    const activeIndex = steps.indexOf(activeStep);
    edge.classList.toggle("active", fromIndex < activeIndex && toIndex === activeIndex);
    edge.classList.toggle("complete", toIndex < activeIndex);
  });

  if (branchLayerEl.dataset.ready === "true") {
    const retrieveIndex = steps.indexOf("retrieve");
    const activeIndex = steps.indexOf(activeStep);
    branchLayerEl.classList.toggle("visible", activeIndex >= retrieveIndex);
    branchLayerEl.classList.toggle("active", activeStep === "retrieve");
    branchLayerEl.classList.toggle("complete", activeIndex > retrieveIndex);
  }

  if (routeVoteLayerEl.dataset.ready === "true") {
    const routeIndex = steps.indexOf("route");
    const activeIndex = steps.indexOf(activeStep);
    routeVoteLayerEl.classList.toggle("visible", activeIndex >= routeIndex);
  }

  if (expertBadgeLayerEl.dataset.ready === "true") {
    const routeIndex = steps.indexOf("route");
    const executeIndex = steps.indexOf("execute");
    const activeIndex = steps.indexOf(activeStep);
    expertBadgeLayerEl.classList.toggle("visible", activeIndex >= routeIndex);
    expertBadgeLayerEl.classList.toggle("active", activeIndex >= routeIndex && activeIndex <= executeIndex);
  }

  if (executeBranchLayerEl.dataset.ready === "true") {
    const executeIndex = steps.indexOf("execute");
    const activeIndex = steps.indexOf(activeStep);
    executeBranchLayerEl.classList.toggle("visible", activeIndex >= executeIndex);
    executeBranchLayerEl.classList.toggle("active", activeStep === "execute");
    executeBranchLayerEl.classList.toggle("complete", activeIndex > executeIndex);
  }

  if (resultRingLayerEl.dataset.ready === "true") {
    const resultIndex = steps.indexOf("result");
    const activeIndex = steps.indexOf(activeStep);
    resultRingLayerEl.classList.toggle("visible", activeIndex >= resultIndex);
    resultRingLayerEl.classList.toggle("active", activeStep === "result");
  }
}

function pushTimeline(title, payload) {
  const item = document.createElement("div");
  item.className = "timeline-item";
  item.dataset.step = title.toLowerCase();
  const heading = document.createElement("h4");
  heading.textContent = title;
  const body = document.createElement("pre");
  body.textContent = JSON.stringify(payload, null, 2);
  const meta = document.createElement("div");
  meta.className = "timeline-meta";
  meta.textContent = stepDescriptions[item.dataset.step] || "";
  item.appendChild(heading);
  item.appendChild(body);
  item.appendChild(meta);
  item.addEventListener("click", () => {
    setActiveDetailStep(item.dataset.step);
  });
  timelineEl.appendChild(item);
  requestAnimationFrame(() => item.classList.add("visible"));
}

function detailForStep(step, payload) {
  if (step === "document") {
    return currentSession
      ? JSON.stringify(
          {
            file_name: currentSession.file_name,
            media_type: currentSession.media_type,
            size_bytes: currentSession.size_bytes,
            preview_text: currentSession.preview_text,
          },
          null,
          2
        )
      : "Upload a document to inspect its parsed content.";
  }
  if (step === "graph") {
    return currentSession
      ? JSON.stringify(
          {
            parse_summary: currentSession.parse_summary,
            graph_summary: currentSession.graph_summary,
          },
          null,
          2
        )
      : "Upload a document to inspect the graph summary.";
  }
  if (!payload) {
    return "Run a query to inspect the pipeline.";
  }
  if (step === "query") {
    return JSON.stringify({ query: payload.query }, null, 2);
  }
  if (step === "retrieve") {
    return JSON.stringify(payload.retrieve, null, 2);
  }
  if (step === "route") {
    return JSON.stringify(payload.route, null, 2);
  }
  if (step === "execute") {
    return JSON.stringify(payload.execute, null, 2);
  }
  if (step === "result") {
    return JSON.stringify(payload.result, null, 2);
  }
  return "Run a query to inspect the pipeline.";
}

function updateStatus({ stage, expert, confidence, note }) {
  if (stage !== undefined) statusStageEl.textContent = stage;
  if (expert !== undefined) statusExpertEl.textContent = expert;
  if (confidence !== undefined) statusConfidenceEl.textContent = confidence;
  if (note !== undefined) statusNoteEl.textContent = note;
}

function setActiveDetailStep(step) {
  activeDetailStep = step;
  document.querySelectorAll(".detail-tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.detailStep === step);
  });
  document.querySelectorAll(".timeline-item").forEach((item) => {
    item.classList.toggle("active", item.dataset.step === step);
  });
  detailsEl.classList.add("flash");
  detailsEl.textContent = detailForStep(step, currentPayload);
  requestAnimationFrame(() => {
    requestAnimationFrame(() => detailsEl.classList.remove("flash"));
  });
}

function bindNodeDetailClicks() {
  document.querySelectorAll(".workflow-node").forEach((node) => {
    node.style.cursor = "pointer";
    node.onclick = () => {
      setActiveDetailStep(node.dataset.step);
    };
  });
}

function bindDetailTabs() {
  detailTabsEl.querySelectorAll(".detail-tab").forEach((button) => {
    button.addEventListener("click", () => {
      setActiveDetailStep(button.dataset.detailStep);
    });
  });
}

function resetView() {
  timelineEl.innerHTML = "";
  currentPayload = null;
  activeDetailStep = currentSession ? "document" : "query";
  detailsEl.textContent = currentSession ? detailForStep("document", null) : "Run a query to inspect the pipeline.";
  updateStatus({
    stage: "Idle",
    expert: "Pending",
    confidence: "--",
    note: currentSession ? "Graph is ready. Ask a numeric question." : "Upload a document to start the workflow.",
  });
  document.querySelectorAll(".workflow-node, .workflow-link").forEach((node) => {
    node.classList.remove("active", "complete");
  });
  branchLinksEl.innerHTML = "";
  branchNodesEl.innerHTML = "";
  branchLayerEl.classList.remove("visible", "active", "complete");
  branchLayerEl.dataset.ready = "false";
  hideHoverCard();
  routeVoteLayerEl.innerHTML = "";
  routeVoteLayerEl.classList.remove("visible");
  routeVoteLayerEl.dataset.ready = "false";
  expertBadgeLabelEl.textContent = "EXPERT";
  expertBadgeLayerEl.classList.remove("visible", "active");
  expertBadgeLayerEl.dataset.ready = "false";
  executeBranchLinksEl.innerHTML = "";
  executeBranchNodesEl.innerHTML = "";
  executeBranchLayerEl.classList.remove("visible", "active", "complete");
  executeBranchLayerEl.dataset.ready = "false";
  resultRingFgEl.setAttribute("stroke-dasharray", "0 100");
  resultRingFgEl.style.stroke = "#78a55a";
  resultRingValueEl.textContent = "Δ 0.00%";
  resultRingLayerEl.classList.remove("visible", "active");
  resultRingLayerEl.dataset.ready = "false";
  setDocumentFlowState(currentSession ? "graph" : null);
  setActiveDetailStep(currentSession ? "document" : "query");
}

function renderExamples(examples) {
  examplesEl.innerHTML = "";
  examples.forEach((example) => {
    const button = document.createElement("button");
    button.className = "chip";
    button.textContent = example;
    button.addEventListener("click", () => {
      queryEl.value = example;
    });
    examplesEl.appendChild(button);
  });
}

async function loadExamples() {
  const response = await fetch("/api/examples");
  const payload = await response.json();
  renderExamples(payload.examples);
}

async function readErrorMessage(response) {
  try {
    const payload = await response.json();
    if (typeof payload.detail === "string" && payload.detail) {
      return payload.detail;
    }
    return JSON.stringify(payload);
  } catch {
    return await response.text();
  }
}

async function fileToBase64(file) {
  const buffer = await file.arrayBuffer();
  let binary = "";
  const bytes = new Uint8Array(buffer);
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.subarray(i, i + chunkSize);
    binary += String.fromCharCode(...chunk);
  }
  return btoa(binary);
}

async function uploadDocument() {
  const file = fileEl.files[0];
  if (!file) return;

  uploadBtn.disabled = true;
  uploadBtn.classList.add("loading");
  currentSession = null;
  setDocumentFlowState("upload");
  updateStatus({
    stage: "Upload",
    expert: "Pending",
    confidence: "--",
    note: "Uploading document and preparing graph build.",
  });

  const contentBase64 = await fileToBase64(file);
  const response = await fetch("/api/document/process", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      file_name: file.name,
      media_type: file.type || "application/octet-stream",
      content_base64: contentBase64,
    }),
  });

  if (!response.ok) {
    const errorText = await readErrorMessage(response);
    uploadBtn.disabled = false;
    uploadBtn.classList.remove("loading");
    updateStatus({
      stage: "Error",
      note: "Document processing failed. Check the uploaded file format.",
    });
    docMetaEl.textContent = errorText;
    return;
  }

  const payload = await response.json();
  currentSession = payload;
  if (payload.query_examples?.length) {
    renderExamples(payload.query_examples);
  }

  docNameEl.textContent = payload.file_name;
  docMetaEl.textContent = `${payload.media_type} · ${payload.size_bytes} bytes`;
  docPreviewEl.textContent = payload.preview_text || "No preview available.";

  const sequence = [
    ["parse", "Parsing document structure."],
    ["extract", "Extracting numeric facts and entities."],
    ["graph", "Graph built and ready for question answering."],
  ];
  for (const [step, note] of sequence) {
    await new Promise((resolve) => setTimeout(resolve, 350));
    setDocumentFlowState(step);
    updateStatus({
      stage: step === "graph" ? "Graph Ready" : step.charAt(0).toUpperCase() + step.slice(1),
      note,
    });
  }

  queryEl.disabled = false;
  runBtn.disabled = false;
  uploadBtn.disabled = false;
  uploadBtn.classList.remove("loading");
  setActiveDetailStep("document");
}

function clearDocument() {
  currentSession = null;
  fileEl.value = "";
  queryEl.disabled = true;
  runBtn.disabled = true;
  docNameEl.textContent = "No document uploaded";
  docMetaEl.textContent = "Upload a document to start the product flow.";
  docPreviewEl.textContent = "Document preview will appear here.";
  setDocumentFlowState(null);
  resetView();
  loadExamples();
}

async function runDemo() {
  const query = queryEl.value.trim();
  if (!query || !currentSession) return;

  runBtn.disabled = true;
  runBtn.classList.add("loading");
  resetView();
  updateStatus({
    stage: "Query",
    expert: "Pending",
    confidence: "--",
    note: "Query captured. Starting retrieval.",
  });
  setStepState("query");
  pushTimeline("Query", { query, note: stepDescriptions.query });
  detailsEl.textContent = "Fetching demo result...";

  const response = await fetch("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, session_id: currentSession.session_id }),
  });
  if (!response.ok) {
    const errorText = await readErrorMessage(response);
    currentPayload = null;
    detailsEl.textContent = errorText;
    updateStatus({
      stage: "Error",
      note: errorText,
    });
    runBtn.disabled = false;
    runBtn.classList.remove("loading");
    return;
  }
  const payload = await response.json();
  currentPayload = payload;

  const sequence = [
    ["retrieve", payload.retrieve],
    ["route", payload.route],
    ["execute", payload.execute],
    ["result", payload.result],
  ];

  for (const [step, content] of sequence) {
    await new Promise((resolve) => setTimeout(resolve, 450));
    if (step === "retrieve") {
      renderRetrieveBranch(payload.retrieve.neighbors, payload.route.selected_expert);
      updateStatus({
        stage: "Retrieve",
        note: `Retrieved ${payload.retrieve.neighbors.length} similar queries.`,
      });
    }
    if (step === "route") {
      renderRouteVotes(payload.route);
      renderExpertBadge(payload.route);
      updateStatus({
        stage: "Route",
        expert: payload.route.selected_expert.toUpperCase(),
        confidence: `${(payload.route.confidence * 100).toFixed(1)}%`,
        note: "Routing vote completed. Expert selected.",
      });
    }
    if (step === "execute") {
      renderExecuteBranch(payload.execute);
      updateStatus({
        stage: "Execute",
        note: `Executing ${payload.route.selected_expert.toUpperCase()} on ${payload.execute.sample_id}.`,
      });
    }
    if (step === "result") {
      renderResultRing(payload.result);
      updateStatus({
        stage: "Result",
        note: `Prediction finished. Absolute error = ${payload.result.abs_error.toFixed(4)}.`,
      });
    }
    setStepState(step);
    pushTimeline(step.charAt(0).toUpperCase() + step.slice(1), content);
    setActiveDetailStep(step);
  }

  runBtn.disabled = false;
  runBtn.classList.remove("loading");
}

runBtn.addEventListener("click", runDemo);
resetBtn.addEventListener("click", resetView);
uploadBtn.addEventListener("click", uploadDocument);
docResetBtn.addEventListener("click", clearDocument);
bindNodeDetailClicks();
bindDetailTabs();
clearDocument();
*/

const STEP_ORDER = ["upload", "query", "extract", "route", "result"];
const STEP_NOTES = {
  upload: "文档已经入库，可以开始按问题抽取图谱。",
  query: "已接收当前问题，准备进入图谱抽取。",
  extract: "大模型已按当前问题抽取所需图谱。",
  route: "系统已完成专家路由与投票选择。",
  result: "专家执行完成，结果已经返回。",
};
const EXPERT_LABELS = {
  sum: "求和专家",
  count: "计数专家",
  predict: "预测专家",
};
const EXTRACTION_MODE_LABELS = {
  llm: "LLM抽取",
  structured_fallback: "结构化回退",
  structured_fallback_after_llm_build_error: "结构化修复",
  pending: "待定",
  ready: "就绪",
  error: "错误",
};
const FACT_LABELS = {
  matched_industry: "命中行业",
  selected_year: "选择年份",
  default_year_applied: "默认年份补全",
  input_record_count: "输入记录数",
  filtered_record_count: "过滤后记录数",
  target_attr: "目标指标",
  ground_truth_available: "是否有真值",
  graph_record_count: "图中记录数",
  extraction_mode: "图谱模式",
  llm_extractor_available: "LLM抽取可用",
  graph_quality_score: "抽图质量",
  expert_ready: "专家输入就绪",
  validation_error_count: "验证错误数",
  validation_warning_count: "验证告警数",
  target_company: "目标公司",
  dataset: "数据来源",
  task_kind: "任务类型",
  sample_id: "样本编号",
};

const queryEl = document.getElementById("query");
const fileEl = document.getElementById("document-file");
const uploadBtn = document.getElementById("upload-btn");
const docResetBtn = document.getElementById("doc-reset-btn");
const docNameEl = document.getElementById("doc-name");
const docMetaEl = document.getElementById("doc-meta");
const runBtn = document.getElementById("run-btn");
const resetBtn = document.getElementById("reset-btn");
const examplesEl = document.getElementById("examples");
const statusStageEl = document.getElementById("status-stage");
const statusExpertEl = document.getElementById("status-expert");
const statusConfidenceEl = document.getElementById("status-confidence");
const statusModeEl = document.getElementById("status-mode");
const statusNoteEl = document.getElementById("status-note");
const evidenceCountEl = document.getElementById("evidence-count");
const recordCountEl = document.getElementById("record-count");
const evidenceLinesEl = document.getElementById("evidence-lines");
const recordListEl = document.getElementById("record-list");
const graphModeBadgeEl = document.getElementById("graph-mode-badge");
const graphStatsBadgeEl = document.getElementById("graph-stats-badge");
const graphSvgEl = document.getElementById("graph-svg");
const graphCaptionEl = document.getElementById("graph-caption");
const resultMetricsEl = document.getElementById("result-metrics");
const routeBarsEl = document.getElementById("route-bars");
const neighborListEl = document.getElementById("neighbor-list");
const executionFactsEl = document.getElementById("execution-facts");
const expertBadgeEl = document.getElementById("expert-badge");
const debugToggleEl = document.getElementById("debug-toggle");
const debugOutputEl = document.getElementById("debug-output");

let currentSession = null;
let currentPayload = null;

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function renderExamples(examples) {
  examplesEl.innerHTML = "";
  examples.forEach((example) => {
    const button = document.createElement("button");
    button.className = "chip";
    button.textContent = example;
    button.addEventListener("click", () => {
      queryEl.value = example;
    });
    examplesEl.appendChild(button);
  });
}

async function loadExamples() {
  const response = await fetch("/api/examples");
  const payload = await response.json();
  renderExamples(payload.examples || []);
}

async function readErrorMessage(response) {
  try {
    const payload = await response.json();
    if (payload && typeof payload.detail === "string") {
      return payload.detail;
    }
    return JSON.stringify(payload);
  } catch {
    return await response.text();
  }
}

async function fileToBase64(file) {
  const buffer = await file.arrayBuffer();
  let binary = "";
  const bytes = new Uint8Array(buffer);
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.subarray(i, i + chunkSize);
    binary += String.fromCharCode(...chunk);
  }
  return btoa(binary);
}

function updateStatus({ stage, expert, confidence, mode, note }) {
  if (stage !== undefined) statusStageEl.textContent = stage;
  if (expert !== undefined) statusExpertEl.textContent = expert;
  if (confidence !== undefined) statusConfidenceEl.textContent = confidence;
  if (mode !== undefined) statusModeEl.textContent = mode;
  if (note !== undefined) statusNoteEl.textContent = note;
}

function setStepState(activeStep) {
  const activeIndex = STEP_ORDER.indexOf(activeStep);
  document.querySelectorAll(".step-chip").forEach((node) => {
    const index = STEP_ORDER.indexOf(node.dataset.step);
    node.classList.toggle("active", index === activeIndex);
    node.classList.toggle("complete", index >= 0 && index < activeIndex);
  });
  document.querySelectorAll(".step-link").forEach((node, index) => {
    node.classList.toggle("active", index + 1 === activeIndex);
    node.classList.toggle("complete", index + 1 < activeIndex);
  });
}

function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "--";
  }
  const number = Number(value);
  const abs = Math.abs(number);
  if (abs >= 1e8) return `${(number / 1e8).toFixed(2)}亿`;
  if (abs >= 1e4) return `${(number / 1e4).toFixed(2)}万`;
  if (abs >= 100) return number.toFixed(1);
  if (abs >= 1) return number.toFixed(2);
  return number.toFixed(4);
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "--";
  }
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function formatExtractionMode(mode) {
  return EXTRACTION_MODE_LABELS[String(mode || "pending").toLowerCase()] || String(mode || "待定");
}

function formatExpert(expert) {
  return EXPERT_LABELS[String(expert || "").toLowerCase()] || "待定";
}

function formatMetricName(name) {
  return FACT_LABELS[name] || String(name).replaceAll("_", " ");
}

function translateGraphLabel(label) {
  const mapping = {
    route: "路由",
    extract: "抽取",
    reads: "读取",
    history: "历史",
  };
  return mapping[label] || label;
}

function translateNodeKind(kind) {
  const mapping = {
    query: "问题节点",
    expert: "专家节点",
    filter: "筛选节点",
    company: "公司节点",
    history: "历史节点",
  };
  return mapping[kind] || kind;
}

function renderMetricCards(payload) {
  resultMetricsEl.innerHTML = "";
  const validation = payload?.graph_extraction?.validation;

  const cards = currentPayload
    ? [
        ["预测值", formatNumber(payload.result.prediction)],
        ["真值", formatNumber(payload.result.answer)],
        ["绝对误差", formatNumber(payload.result.abs_error)],
        ["专家", formatExpert(payload.route.selected_expert)],
        ["置信度", formatPercent(payload.route.confidence)],
        ["图谱模式", formatExtractionMode(payload.execute.graph_build.extraction_mode)],
        ["抽图质量", validation ? formatPercent(validation.quality_score) : "--"],
      ]
    : [
        ["预测值", "--"],
        ["真值", "--"],
        ["绝对误差", "--"],
        ["专家", "待定"],
        ["置信度", "--"],
        ["图谱模式", "待定"],
        ["抽图质量", "--"],
      ];

  cards.forEach(([label, value]) => {
    const card = document.createElement("div");
    card.className = "metric-card";
    const labelEl = document.createElement("span");
    labelEl.className = "metric-label";
    labelEl.textContent = label;
    const valueEl = document.createElement("strong");
    valueEl.className = "metric-value";
    valueEl.textContent = value;
    card.append(labelEl, valueEl);
    resultMetricsEl.appendChild(card);
  });
}

function renderRouteBars(route) {
  routeBarsEl.innerHTML = "";
  const experts = ["sum", "count", "predict"];
  const maxScore = Math.max(
    1e-6,
    ...experts.map((expert) => Number(route?.expert_scores?.[expert] || 0))
  );
  experts.forEach((expert) => {
    const score = Number(route?.expert_scores?.[expert] || 0);
    const row = document.createElement("div");
    row.className = "route-row";

    const label = document.createElement("div");
    label.className = "route-label";
    label.textContent = formatExpert(expert);

    const track = document.createElement("div");
    track.className = "route-track";
    const fill = document.createElement("div");
    fill.className = "route-fill";
    fill.style.width = `${(score / maxScore) * 100}%`;
    fill.style.background =
      expert === "sum" ? "var(--sum)" : expert === "count" ? "var(--count)" : "var(--predict)";
    track.appendChild(fill);

    const value = document.createElement("div");
    value.className = "route-score";
    value.textContent = score ? score.toFixed(2) : "0";

    row.append(label, track, value);
    routeBarsEl.appendChild(row);
  });
}

function renderNeighborList(payload) {
  neighborListEl.innerHTML = "";
  const neighbors = payload?.retrieve?.neighbors || [];
  if (!neighbors.length) {
    const empty = document.createElement("div");
    empty.className = "neighbor-card";
    empty.textContent = "运行一次查询后，这里会展示影响路由决策的相似问题。";
    neighborListEl.appendChild(empty);
    return;
  }
  neighbors.slice(0, 4).forEach((neighbor, index) => {
    const card = document.createElement("div");
    card.className = "neighbor-card";
    const title = document.createElement("p");
    title.className = "neighbor-title";
    title.textContent = `#${index + 1} · ${formatExpert(neighbor.expert)} · ${neighbor.score.toFixed(3)}`;
    const meta = document.createElement("p");
    meta.className = "neighbor-meta";
    meta.textContent = neighbor.text;
    card.append(title, meta);
    neighborListEl.appendChild(card);
  });
}

function renderExecutionFacts(payload) {
  executionFactsEl.innerHTML = "";
  const execute = payload?.execute;
  if (!execute) {
    const empty = document.createElement("div");
    empty.className = "fact-card";
    empty.textContent = "执行一次查询后，这里会展示模型、图规模和图谱构建细节。";
    executionFactsEl.appendChild(empty);
    return;
  }

  const facts = [
    ["模型文件", execute.artifact.checkpoint_path.split("/").pop()],
    ["数据来源", execute.dataset],
    ["任务类型", execute.task_kind],
    ["样本编号", execute.sample_id],
    ["图规模", `${execute.num_nodes} 个节点 / ${execute.num_edges} 条边`],
  ];
  const validation = execute.graph_extraction?.validation;
  if (validation) {
    facts.push(
      ["抽图就绪", validation.expert_ready ? "是" : "否"],
      ["抽图质量", formatPercent(validation.quality_score)],
      ["验证告警", `${validation.warnings.length} 条`],
      ["验证错误", `${validation.errors.length} 条`]
    );
  }
  Object.entries(execute.graph_build || {}).forEach(([key, value]) => {
    if (typeof value === "object") {
      return;
    }
    facts.push([formatMetricName(key), String(value)]);
  });

  facts.forEach(([key, value]) => {
    const card = document.createElement("div");
    card.className = "fact-card";
    const title = document.createElement("p");
    title.className = "fact-key";
    title.textContent = key;
    const meta = document.createElement("p");
    meta.className = "fact-value";
    meta.textContent = value;
    card.append(title, meta);
    executionFactsEl.appendChild(card);
  });
}

function renderEvidence(payload) {
  evidenceLinesEl.innerHTML = "";
  recordListEl.innerHTML = "";

  const evidence = payload?.evidence;
  const lines = evidence
    ? evidence.lines
    : (currentSession?.document_text || "")
        .split(/\r?\n/)
        .filter((line) => line.trim())
        .map((line, index) => ({ index: index + 1, text: line, relevant: false, reasons: [] }));

  evidenceCountEl.textContent = `${lines.filter((line) => line.relevant).length} 条相关证据`;

  lines.forEach((line) => {
    const card = document.createElement("div");
    card.className = `evidence-line${line.relevant ? " relevant" : ""}`;

    const meta = document.createElement("div");
    meta.className = "line-meta";
    meta.innerHTML = `<span>第 ${line.index} 行</span><span>${line.relevant ? "进入图谱" : "上下文"}</span>`;

    const body = document.createElement("p");
    body.className = "line-text";
    body.textContent = line.text;
    card.append(meta, body);

    if (line.reasons?.length) {
      const reasons = document.createElement("div");
      reasons.className = "line-reasons";
      line.reasons.forEach((reason) => {
        const pill = document.createElement("span");
        pill.className = "reason-pill";
        pill.textContent = reason;
        reasons.appendChild(pill);
      });
      card.appendChild(reasons);
    }
    evidenceLinesEl.appendChild(card);
  });

  const records = evidence?.records || [];
  recordCountEl.textContent = `${records.length} 条抽取记录`;
  if (!records.length) {
    const empty = document.createElement("div");
    empty.className = "record-card";
    empty.textContent = currentSession
      ? "运行一次查询后，这里会展示按问题抽取出的规范记录。"
      : "上传文档后，这里会展示与问题相关的抽取记录。";
    recordListEl.appendChild(empty);
    return;
  }

  records.forEach((record) => {
    const card = document.createElement("div");
    card.className = `record-card${record.in_graph ? " in-graph" : ""}${record.is_target ? " is-target" : ""}`;

    const title = document.createElement("p");
    title.className = "record-title";
    title.textContent = record.company_name;

    const meta = document.createElement("p");
    meta.className = "record-meta";
    meta.textContent = [
      record.industry || "行业未知",
      record.year || "年份未知",
      `收入 ${formatNumber(record.revenue)}`,
      record.operating_profit !== null && record.operating_profit !== undefined
        ? `营业利润 ${formatNumber(record.operating_profit)}`
        : null,
      record.net_profit !== null && record.net_profit !== undefined
        ? `净利润 ${formatNumber(record.net_profit)}`
        : null,
      record.employees !== null && record.employees !== undefined
        ? `员工数 ${formatNumber(record.employees)}`
        : null,
    ]
      .filter(Boolean)
      .join(" · ");

    const tags = document.createElement("div");
    tags.className = "record-tags";
    if (record.in_graph) {
      const tag = document.createElement("span");
      tag.className = "tag";
      tag.textContent = "已进入图谱";
      tags.appendChild(tag);
    }
    if (record.is_target) {
      const tag = document.createElement("span");
      tag.className = "tag";
      tag.textContent = "目标公司";
      tags.appendChild(tag);
    }
    if (record.is_history) {
      const tag = document.createElement("span");
      tag.className = "tag";
      tag.textContent = "历史记录";
      tags.appendChild(tag);
    }

    card.append(title, meta, tags);
    recordListEl.appendChild(card);
  });
}

function svgEl(name, attributes = {}) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.entries(attributes).forEach(([key, value]) => {
    node.setAttribute(key, String(value));
  });
  return node;
}

function nodePalette(accent) {
  const palette = {
    query: { fill: "#e8f6f6", stroke: "var(--accent)" },
    expert: { fill: "#e8eff9", stroke: "#2f5f8f" },
    filter: { fill: "#eef8f8", stroke: "#5aa9ae" },
    company: { fill: "#f4faf9", stroke: "#bfd6d3" },
    target: { fill: "#fcf4df", stroke: "var(--target)" },
    history: { fill: "#eef2f4", stroke: "var(--history)" },
    sum: { fill: "#e3f4f4", stroke: "var(--sum)" },
    count: { fill: "#edf8f2", stroke: "var(--count)" },
    predict: { fill: "#eaf0fb", stroke: "var(--predict)" },
  };
  return palette[accent] || palette.company;
}

function layoutGraphNodes(graphView) {
  const positions = {};
  const filters = graphView.nodes.filter((node) => node.kind === "filter");
  const histories = graphView.nodes.filter((node) => node.kind === "history");
  const companies = graphView.nodes.filter((node) => node.kind === "company");

  positions.query = { x: 380, y: 60, width: 250, height: 56 };
  positions.expert = { x: 380, y: 210, width: 150, height: 54 };

  filters.forEach((node, index) => {
    const spacing = 560 / Math.max(filters.length, 1);
    positions[node.id] = {
      x: 110 + spacing * index,
      y: 130,
      width: 132,
      height: 42,
    };
  });

  histories.forEach((node, index) => {
    positions[node.id] = {
      x: 110,
      y: 260 + index * 68,
      width: 150,
      height: 42,
    };
  });

  companies.forEach((node, index) => {
    const spacing = 560 / Math.max(companies.length - 1, 1);
    positions[node.id] = {
      x: companies.length === 1 ? 380 : 100 + spacing * index,
      y: 350,
      width: 148,
      height: 58,
    };
  });

  return positions;
}

function renderGraph(payload) {
  graphSvgEl.innerHTML = "";

  if (!payload?.graph_view) {
    graphModeBadgeEl.textContent = "未生成";
    graphStatsBadgeEl.textContent = "0 个节点 / 0 条边";
    graphCaptionEl.textContent = currentSession
      ? "输入一个问题后，这里会按当前问题生成图谱。"
      : "上传一份文档后，这里会显示查询驱动的图谱。";
    const text = svgEl("text", {
      x: 380,
      y: 210,
      "text-anchor": "middle",
      fill: "#617380",
      "font-size": 18,
    });
    text.textContent = currentSession
      ? "运行一次查询后生成图谱。"
      : "文档证据和图谱会显示在这里。";
    graphSvgEl.appendChild(text);
    return;
  }

  const graphView = payload.graph_view;
  graphModeBadgeEl.textContent = formatExtractionMode(payload.execute.graph_build.extraction_mode);
  graphStatsBadgeEl.textContent = `${graphView.stats.num_nodes} 个节点 / ${graphView.stats.num_edges} 条边`;
  graphCaptionEl.textContent = `当前问题图谱包含 ${graphView.stats.num_nodes} 个公司节点，图谱生成模式为 ${formatExtractionMode(graphView.stats.mode)}。`;

  const positions = layoutGraphNodes(graphView);
  graphView.edges.forEach((edge) => {
    const source = positions[edge.source];
    const target = positions[edge.target];
    if (!source || !target) return;
    const line = svgEl("line", {
      x1: source.x,
      y1: source.y + source.height / 2,
      x2: target.x,
      y2: target.y - target.height / 2,
      class: "graph-edge",
    });
    graphSvgEl.appendChild(line);
    const label = svgEl("text", {
      x: (source.x + target.x) / 2,
      y: (source.y + target.y) / 2 - 8,
      "text-anchor": "middle",
      class: "graph-edge-label",
    });
    label.textContent = translateGraphLabel(edge.label);
    graphSvgEl.appendChild(label);
  });

  graphView.nodes.forEach((node) => {
    const position = positions[node.id];
    if (!position) return;
    const { fill, stroke } = nodePalette(node.accent || node.kind);
    const group = svgEl("g");
    const rect = svgEl("rect", {
      x: position.x - position.width / 2,
      y: position.y - position.height / 2,
      width: position.width,
      height: position.height,
      rx: 18,
      ry: 18,
      fill,
      stroke,
      class: "graph-node-card",
    });
    const title = svgEl("text", {
      x: position.x,
      y: position.y - 4,
      "text-anchor": "middle",
      class: "graph-node-title",
    });
    title.textContent = node.label.length > 28 ? `${node.label.slice(0, 27)}…` : node.label;
    const subtitle = svgEl("text", {
      x: position.x,
      y: position.y + 14,
      "text-anchor": "middle",
      class: "graph-node-subtitle",
    });
    subtitle.textContent = node.subtitle || translateNodeKind(node.kind);
    group.append(rect, title, subtitle);
    graphSvgEl.appendChild(group);
  });
}

function renderExpert(payload) {
  renderMetricCards(payload);
  renderRouteBars(payload?.route || {});
  renderNeighborList(payload);
  renderExecutionFacts(payload);

  if (!payload) {
    expertBadgeEl.textContent = "待定";
    return;
  }
  expertBadgeEl.textContent = formatExpert(payload.route.selected_expert);
  expertBadgeEl.style.background =
    payload.route.selected_expert === "sum"
      ? "#fff0e2"
      : payload.route.selected_expert === "count"
        ? "#e8f0f0"
        : "#ebf2fd";
  expertBadgeEl.style.color =
    payload.route.selected_expert === "sum"
      ? "var(--sum)"
      : payload.route.selected_expert === "count"
        ? "var(--count)"
        : "var(--predict)";
}

function renderDebug(payload) {
  debugOutputEl.textContent = payload
    ? JSON.stringify(payload, null, 2)
    : "运行一次查询后，这里会显示原始返回数据。";
}

function renderWorkbench() {
  renderEvidence(currentPayload);
  renderGraph(currentPayload);
  renderExpert(currentPayload);
  renderDebug(currentPayload);
}

function resetView() {
  currentPayload = null;
  setStepState(currentSession ? "upload" : null);
  updateStatus({
    stage: currentSession ? "文档就绪" : "空闲",
    expert: "待定",
    confidence: "--",
    mode: currentSession ? "就绪" : "待定",
    note: currentSession ? STEP_NOTES.upload : "上传一份文档后开始体验整个分析流程。",
  });
  renderWorkbench();
}

function clearDocument() {
  currentSession = null;
  currentPayload = null;
  fileEl.value = "";
  queryEl.value = "";
  queryEl.disabled = true;
  runBtn.disabled = true;
  docNameEl.textContent = "尚未上传文档";
  docMetaEl.textContent = "支持上传小型报告、备忘录、CSV、JSON 或 Markdown 表格。";
  renderExamples([]);
  resetView();
  loadExamples();
}

async function uploadDocument() {
  const file = fileEl.files[0];
  if (!file) return;

  uploadBtn.disabled = true;
  uploadBtn.classList.add("loading");
  currentSession = null;
  currentPayload = null;
  setStepState("upload");
  updateStatus({
    stage: "上传中",
    expert: "待定",
    confidence: "--",
    mode: "待定",
    note: "正在上传文档并准备分析工作台。",
  });

  const contentBase64 = await fileToBase64(file);
  const response = await fetch("/api/document/process", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      file_name: file.name,
      media_type: file.type || "application/octet-stream",
      content_base64: contentBase64,
    }),
  });

  if (!response.ok) {
    const errorText = await readErrorMessage(response);
    uploadBtn.disabled = false;
    uploadBtn.classList.remove("loading");
    updateStatus({
      stage: "错误",
      mode: "错误",
      note: errorText,
    });
    docMetaEl.textContent = errorText;
    return;
  }

  currentSession = await response.json();
  docNameEl.textContent = currentSession.file_name;
  docMetaEl.textContent = `${currentSession.media_type} · ${currentSession.size_bytes} 字节 · ${currentSession.parse_summary.parsed_record_count || 0} 条解析记录`;
  queryEl.disabled = false;
  runBtn.disabled = false;
  renderExamples(currentSession.query_examples || []);
  uploadBtn.disabled = false;
  uploadBtn.classList.remove("loading");
  resetView();
}

async function runDemo() {
  const query = queryEl.value.trim();
  if (!query || !currentSession) return;

  runBtn.disabled = true;
  runBtn.classList.add("loading");
  currentPayload = null;
  setStepState("query");
  updateStatus({
    stage: "问题接收",
    expert: "待定",
    confidence: "--",
    mode: "待定",
    note: STEP_NOTES.query,
  });
  renderWorkbench();

  const response = await fetch("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, session_id: currentSession.session_id }),
  });

  if (!response.ok) {
    const errorText = await readErrorMessage(response);
    updateStatus({
      stage: "错误",
      mode: "错误",
      note: errorText,
    });
    debugOutputEl.textContent = errorText;
    runBtn.disabled = false;
    runBtn.classList.remove("loading");
    return;
  }

  currentPayload = await response.json();

  await sleep(180);
  setStepState("extract");
  updateStatus({
    stage: "图谱抽取",
    mode: formatExtractionMode(currentPayload.execute.graph_build.extraction_mode),
    note: STEP_NOTES.extract,
  });
  renderWorkbench();

  await sleep(180);
  setStepState("route");
  updateStatus({
    stage: "专家路由",
    expert: formatExpert(currentPayload.route.selected_expert),
    confidence: formatPercent(currentPayload.route.confidence),
    mode: formatExtractionMode(currentPayload.execute.graph_build.extraction_mode),
    note: STEP_NOTES.route,
  });
  renderWorkbench();

  await sleep(180);
  setStepState("result");
  updateStatus({
    stage: "结果返回",
    expert: formatExpert(currentPayload.route.selected_expert),
    confidence: formatPercent(currentPayload.route.confidence),
    mode: formatExtractionMode(currentPayload.execute.graph_build.extraction_mode),
    note: `${STEP_NOTES.result} 当前绝对误差为 ${formatNumber(currentPayload.result.abs_error)}。`,
  });
  renderWorkbench();

  runBtn.disabled = false;
  runBtn.classList.remove("loading");
}

debugToggleEl.addEventListener("click", () => {
  debugOutputEl.hidden = !debugOutputEl.hidden;
});
runBtn.addEventListener("click", runDemo);
resetBtn.addEventListener("click", resetView);
uploadBtn.addEventListener("click", uploadDocument);
docResetBtn.addEventListener("click", clearDocument);

clearDocument();
