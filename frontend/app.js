(() => {
  "use strict";

  const API_BASE = (window.MEDICAL_API_BASE || "http://127.0.0.1:8000").replace(/\/$/, "");
  const API = {
    summary: "/api/b1/datos/resumen",
    statistics: "/api/b1/datos/estadisticas",
    clusteringMetrics: "/api/b1/clustering/jerarquico",
    clusteringProfiles: "/api/b1/clustering/perfiles",
    classification: "/api/b1/clasificacion/resumen",
    prediction: "/api/b1/clasificacion/predecir",
  };
  const numberFormat = new Intl.NumberFormat("es-CO");
  const tabTitles = {
    overview: "Resumen",
    clustering: "Clustering",
    classification: "Clasificación",
    prediction: "Demo predictiva",
  };

  const $ = (selector, root = document) => root.querySelector(selector);
  const formatNumber = (value, digits = 2) => {
    const number = Number(value);
    return Number.isFinite(number)
      ? new Intl.NumberFormat("es-CO", { maximumFractionDigits: digits }).format(number)
      : "—";
  };

  async function request(path, options = {}) {
    const response = await fetch(`${API_BASE}${path}`, {
      ...options,
      headers: {
        Accept: "application/json",
        ...(options.body ? { "Content-Type": "application/json" } : {}),
        ...options.headers,
      },
    });

    if (!response.ok) {
      let detail = "";
      try {
        const body = await response.json();
        detail = body.detail || body.message || "";
      } catch {
        detail = "";
      }
      throw new Error(detail || `La API respondió con estado ${response.status}.`);
    }

    return response.json();
  }

  function setApiStatus(status, message) {
    const indicator = $("#apiIndicator");
    indicator.classList.remove("connected", "disconnected");
    if (status) indicator.classList.add(status);
    $("#apiStatusText").textContent = message;
  }

  function showError(element, error, context) {
    element.hidden = false;
    element.textContent = `${context}: ${error.message} Comprueba que FastAPI esté ejecutándose en ${API_BASE} y permita solicitudes desde el origen del frontend.`;
  }

  function clearError(element) {
    element.hidden = true;
    element.textContent = "";
  }

  function renderDistribution(distribution) {
    const chart = $("#distributionChart");
    const entries = Object.entries(distribution || {});
    if (!entries.length) {
      chart.innerHTML = '<div class="empty-state"><span>◷</span><b>Sin distribución disponible</b><small>La API no reportó etiquetas para la variable objetivo.</small></div>';
      return;
    }

    const maxCount = Math.max(...entries.map(([, count]) => Number(count) || 0), 1);
    chart.innerHTML = entries.map(([label, count]) => {
      const value = Number(count) || 0;
      const width = Math.max(0, Math.min(100, (value / maxCount) * 100));
      return `<div class="distribution-item">
        <span class="distribution-label" title="${escapeHtml(label)}">${escapeHtml(label)}</span>
        <span class="distribution-track"><span class="distribution-fill" style="width:${width}%"></span></span>
        <span class="distribution-count">${numberFormat.format(value)}</span>
      </div>`;
    }).join("");
  }

  function renderSummary(summary) {
    $("#rowCount").textContent = numberFormat.format(summary.filas ?? 0);
    $("#columnCount").textContent = numberFormat.format((summary.columnas || []).length);
    const missing = Object.values(summary.nulos || {}).reduce((total, count) => total + (Number(count) || 0), 0);
    $("#missingCount").textContent = numberFormat.format(missing);
    $("#duplicateCount").textContent = numberFormat.format(summary.duplicados ?? 0);
    $("#targetName").textContent = summary.objetivo || "—";
    $("#qualityMissing").textContent = numberFormat.format(missing);
    $("#qualityDuplicates").textContent = numberFormat.format(summary.duplicados ?? 0);
    $("#qualityTarget").textContent = summary.objetivo || "—";
    renderDistribution(summary.distribucion_objetivo);
  }

  function renderStatistics(statistics) {
    const table = $("#statsTable");
    const rows = Object.entries(statistics || {});
    if (!rows.length) {
      table.innerHTML = '<tr><td colspan="6"><div class="empty-state"><span>◷</span><b>Sin estadísticas disponibles</b><small>La API no devolvió variables descriptivas.</small></div></td></tr>';
      return;
    }

    table.innerHTML = rows.map(([name, values]) => `<tr>
      <td class="variable-cell">${escapeHtml(name)}</td>
      <td class="numeric-cell">${formatNumber(values.min)}</td>
      <td class="numeric-cell">${formatNumber(values.mediana)}</td>
      <td class="numeric-cell">${formatNumber(values.media)}</td>
      <td class="numeric-cell">${formatNumber(values.max)}</td>
      <td class="numeric-cell">${formatNumber(values.asimetria)}</td>
    </tr>`).join("");
  }

  async function loadOverview() {
    const errorBox = $("#overviewError");
    clearError(errorBox);
    setApiStatus("", "Conectando API");

    const results = await Promise.allSettled([
      request(API.summary),
      request(API.statistics),
    ]);
    const summaryResult = results[0];
    const statisticsResult = results[1];

    if (summaryResult.status === "fulfilled") {
      renderSummary(summaryResult.value);
    } else {
      $("#distributionChart").innerHTML = '<div class="empty-state"><span>!</span><b>Resumen no disponible</b><small>Revisa la conexión con el backend.</small></div>';
      showError(errorBox, summaryResult.reason, "No fue posible cargar el resumen");
    }

    if (statisticsResult.status === "fulfilled") {
      renderStatistics(statisticsResult.value);
    } else {
      $("#statsTable").innerHTML = '<tr><td colspan="6"><div class="empty-state"><span>!</span><b>Estadísticas no disponibles</b><small>Revisa la conexión con el backend.</small></div></td></tr>';
      if (summaryResult.status === "fulfilled") {
        showError(errorBox, statisticsResult.reason, "No fue posible cargar las estadísticas");
      }
    }

    if (summaryResult.status === "fulfilled" || statisticsResult.status === "fulfilled") {
      setApiStatus("connected", "API conectada");
    } else {
      setApiStatus("disconnected", "API sin conexión");
    }
  }

  async function loadClustering() {
    const results = await Promise.allSettled([
      request(API.clusteringMetrics),
      request(`${API.clusteringProfiles}?metodo=kmeans&k=5`),
    ]);

    if (results[0].status === "fulfilled") {
      renderClusteringMetrics(results[0].value);
      $("#clusteringNotice").hidden = true;
    }
    if (results[1].status === "fulfilled") {
      renderClusterProfiles(results[1].value);
      $("#clusteringNotice").hidden = true;
    }
  }

  function renderClusteringMetrics(data) {
    const rows = Array.isArray(data) ? data : data.resultados;
    if (!Array.isArray(rows) || rows.length === 0) return;
    $("#clusteringMetrics").innerHTML = rows.map((row) => `<tr>
      <td class="method-cell">${escapeHtml(row.metodo)}</td>
      <td>${escapeHtml(row.k)}</td>
      <td>${formatNumber(row.silhouette, 4)}</td>
      <td>${formatNumber(row.cofenetica, 4)}</td>
      <td>${formatNumber(row.tamano_min, 0)}</td>
    </tr>`).join("");
  }

  function renderClusterProfiles(data) {
    const rows = Array.isArray(data) ? data : data.perfiles;
    if (!Array.isArray(rows) || rows.length === 0) return;
    $("#clusterProfiles").innerHTML = rows.map((row, index) => `<tr>
      <td><span class="cluster-chip">${escapeHtml(row.cluster ?? index + 1)}</span></td>
      <td>${formatNumber(row.pacientes, 0)}</td>
      <td>${formatNumber(row.edad)}</td>
      <td>${formatNumber(row.pulso)}</td>
      <td>${formatNumber(row.sistolica)}</td>
      <td>${formatNumber(row.glucosa)}</td>
      <td>${formatNumber(row.ckmb_mediana)}</td>
      <td>${formatNumber(row.troponina_mediana)}</td>
      <td>${formatNumber(row.pct_infarto)}%</td>
    </tr>`).join("");
  }

  async function loadClassification() {
    try {
      const data = await request(API.classification);
      renderClassification(data);
      $("#classificationNotice").hidden = true;
    } catch {
      // The endpoint is not part of the current backend; the notice explains this state.
    }
  }

  function renderClassification(data) {
    if (data.accuracy !== undefined) $("#modelAccuracy").textContent = `${formatNumber(Number(data.accuracy) * 100, 2)}%`;
    const matrix = data.confusion_matrix;
    if (Array.isArray(matrix)) renderConfusionMatrix(matrix, data.clases || ["Negativo", "Positivo"]);
    const importances = data.feature_importances;
    if (importances) renderImportances(importances);
    if (data.classification_report) renderClassificationReport(data.classification_report);
  }

  function renderConfusionMatrix(matrix, classes) {
    if (!matrix.length || !Array.isArray(matrix[0])) return;
    const columnHeadings = classes.map((name) => `<span class="matrix-axis">${escapeHtml(name)}</span>`).join("");
    const cells = matrix.map((row, rowIndex) => `<span class="matrix-row-label">${escapeHtml(classes[rowIndex] || `Clase ${rowIndex}`)}</span>${row.map((value, columnIndex) => `<span class="matrix-cell${rowIndex === columnIndex ? "" : " off-diagonal"}">${formatNumber(value, 0)}</span>`).join("")}`).join("");
    $("#confusionMatrix").innerHTML = `<div class="matrix-layout"><span></span>${columnHeadings}${cells}</div>`;
  }

  function renderImportances(raw) {
    const entries = Array.isArray(raw)
      ? raw.map((item) => [item.variable ?? item.feature, item.importance ?? item.valor])
      : Object.entries(raw);
    const usable = entries.filter(([name, value]) => name !== undefined && Number.isFinite(Number(value)));
    if (!usable.length) return;
    const max = Math.max(...usable.map(([, value]) => Number(value)), 1e-12);
    $("#predictorCount").textContent = numberFormat.format(usable.length);
    $("#importanceChart").innerHTML = usable.map(([name, value]) => {
      const importance = Number(value);
      return `<div class="importance-row"><span class="importance-label">${escapeHtml(name)}</span><span class="importance-track"><span class="importance-fill" style="width:${Math.max(0, Math.min(100, importance / max * 100))}%"></span></span><span class="importance-value">${formatNumber(importance, 3)}</span></div>`;
    }).join("");
  }

  function renderClassificationReport(report) {
    if (typeof report === "string") {
      $("#classificationReport").innerHTML = `<tr><td colspan="5"><pre class="report-text">${escapeHtml(report)}</pre></td></tr>`;
      return;
    }
    const rows = Object.entries(report).filter(([, values]) => values && typeof values === "object");
    if (!rows.length) return;
    $("#classificationReport").innerHTML = rows.map(([label, values]) => `<tr>
      <td class="report-class">${escapeHtml(label)}</td>
      <td>${formatNumber(values.precision)}</td>
      <td>${formatNumber(values.recall)}</td>
      <td>${formatNumber(values["f1-score"] ?? values.f1_score)}</td>
      <td>${formatNumber(values.support, 0)}</td>
    </tr>`).join("");
  }

  async function submitPrediction(event) {
    event.preventDefault();
    const form = event.currentTarget;
    if (!form.reportValidity()) return;

    const button = $("#predictButton");
    const errorBox = $("#predictionError");
    const result = $("#predictionResult");
    clearError(errorBox);
    button.disabled = true;
    button.innerHTML = '<span class="loader"></span> Consultando modelo…';

    const formData = new FormData(form);
    const patient = Object.fromEntries([...formData.entries()].map(([key, value]) => [key, Number(value)]));

    try {
      const data = await request(API.prediction, {
        method: "POST",
        body: JSON.stringify(patient),
      });
      renderPrediction(data, result);
    } catch (error) {
      showError(errorBox, error, "No fue posible obtener la predicción");
      result.className = "surface-card result-card";
      result.innerHTML = `<div class="result-placeholder"><div class="result-illustration"><span class="result-ring"></span><span class="result-heart">!</span></div><span class="result-eyebrow">SERVICIO NO DISPONIBLE</span><h3>No se pudo consultar<br />el modelo</h3><p>El backend aún no tiene disponible la ruta de predicción o no responde.</p></div>`;
    } finally {
      button.disabled = false;
      button.innerHTML = 'Consultar al modelo <span aria-hidden="true">→</span>';
    }
  }

  function renderPrediction(data, container) {
    const rawPrediction = data.prediction ?? data.prediccion ?? data.result;
    if (rawPrediction === undefined || rawPrediction === null) {
      throw new Error("La respuesta no contiene el campo prediction.");
    }
    const normalized = String(rawPrediction).toLowerCase();
    const positive = ["positive", "positivo", "1", "true", "alto"].includes(normalized);
    const label = positive ? "Riesgo detectado" : "Sin riesgo detectado";
    const probability = data.probability ?? data.probabilidad;
    const probabilityMarkup = probability === undefined
      ? ""
      : `<strong class="probability-value">${formatNumber(Number(probability) <= 1 ? Number(probability) * 100 : probability, 1)}%</strong><span class="probability-caption">Confianza de la clase predicha</span>`;

    container.className = `surface-card result-card ${positive ? "result-positive" : "result-negative"}`;
    container.innerHTML = `<div><span class="result-outcome-icon">${positive ? "!" : "✓"}</span><span class="result-outcome-label">RESULTADO DEL MODELO</span><h3>${label}</h3><p>${positive ? "La salida del modelo sugiere una clase positiva. Solicita valoración de un profesional de la salud." : "La salida del modelo corresponde a una clase negativa. Esto no descarta una condición clínica."}</p>${probabilityMarkup}</div>`;
  }

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, (character) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    })[character]);
  }

  function switchTab(tabId) {
    if (!tabTitles[tabId]) return;
    document.querySelectorAll(".page-section").forEach((section) => section.classList.toggle("active", section.id === tabId));
    document.querySelectorAll(".nav-item").forEach((button) => button.classList.toggle("active", button.dataset.tab === tabId));
    $("#breadcrumbCurrent").textContent = tabTitles[tabId];
    history.replaceState(null, "", `#${tabId}`);
    $("#sidebar").classList.remove("open");
    $("#sidebarScrim").classList.remove("visible");
    if (tabId === "clustering") loadClustering();
    if (tabId === "classification") loadClassification();
  }

  function initTheme() {
    const savedTheme = localStorage.getItem("cardiolab-theme");
    if (savedTheme === "dark") document.body.classList.add("dark-theme");
    $("#themeToggle").addEventListener("click", () => {
      document.body.classList.toggle("dark-theme");
      localStorage.setItem("cardiolab-theme", document.body.classList.contains("dark-theme") ? "dark" : "light");
    });
  }

  function init() {
    document.querySelectorAll("[data-tab]").forEach((button) => {
      button.addEventListener("click", () => switchTab(button.dataset.tab));
    });
    document.querySelectorAll("[data-open-tab]").forEach((button) => {
      button.addEventListener("click", () => switchTab(button.dataset.openTab));
    });
    $("#refreshButton").addEventListener("click", loadOverview);
    $("#predictionForm").addEventListener("submit", submitPrediction);
    $("#dendrogramImage").addEventListener("error", () => {
      $("#dendrogramImage").hidden = true;
      $("#dendrogramFallback").hidden = false;
    });
    $("#menuToggle").addEventListener("click", () => {
      $("#sidebar").classList.add("open");
      $("#sidebarScrim").classList.add("visible");
    });
    $("#sidebarScrim").addEventListener("click", () => {
      $("#sidebar").classList.remove("open");
      $("#sidebarScrim").classList.remove("visible");
    });
    initTheme();
    loadOverview();

    const initialTab = location.hash.slice(1);
    if (tabTitles[initialTab]) switchTab(initialTab);
  }

  document.addEventListener("DOMContentLoaded", init);
})();
