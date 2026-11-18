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
    clusterAssignment: "/api/b1/clustering/asignar",
    dbscan: "/api/b1/dbscan/analisis",
    regression: "/api/b2/regresion/resumen",
    mdp: "/api/b2/mdp/resumen",
    rl: "/api/b2/rl/resumen",
  };
  const CLUSTER_DISCLAIMER = "Agrupación no supervisada con fines exploratorios. El porcentaje corresponde a los pacientes del dataset de entrenamiento en ese grupo, no a la probabilidad de este paciente. No es un diagnóstico.";
  const numberFormat = new Intl.NumberFormat("es-CO");
  let activeTreeView = null;
  const tabTitles = {
    intro: "Introducción",
    overview: "Resumen",
    clustering: "Clustering",
    classification: "Clasificación",
    prediction: "Consulta de clasificación",
    regression: "Regresión",
    mdp: "MDP · Iteración de valor",
    rl: "Q-Learning y DQN",
    conclusions: "Conclusiones",
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
        const responseDetail = body.detail ?? body.message;
        if (Array.isArray(responseDetail)) {
          detail = responseDetail
            .map((item) => typeof item === "string" ? item : item?.msg)
            .filter(Boolean)
            .join(" ");
        } else if (typeof responseDetail === "string") {
          detail = responseDetail;
        }
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

  async function loadDbscanAnalysis() {
    const form = $("#dbscanForm");
    const button = $("#dbscanSubmit");
    const status = $("#dbscanStatus");
    const parameters = new FormData(form);
    const query = new URLSearchParams({
      eps: parameters.get("eps"),
      min_samples: parameters.get("min_samples"),
    });

    button.disabled = true;
    status.hidden = false;
    status.textContent = "Calculando el perfil DBSCAN en el backend…";
    try {
      const data = await request(`${API.dbscan}?${query}`);
      renderDbscanAnalysis(data);
      status.textContent = "Análisis actualizado con datos del backend.";
    } catch (error) {
      status.textContent = `No fue posible cargar el análisis DBSCAN: ${error.message}`;
    } finally {
      button.disabled = false;
    }
  }

  function renderDbscanAnalysis(data) {
    $("#dbscanTotal").textContent = numberFormat.format(data.total_pacientes);
    $("#dbscanNoise").textContent = numberFormat.format(data.pacientes_ruido);
    $("#dbscanOutlierPct").textContent = `${formatNumber(data.porcentaje_atipicos)}%`;
    $("#dbscanDense").textContent = numberFormat.format(data.pacientes_densos);
    $("#dbscanParameters").textContent =
      `eps = ${formatNumber(data.parametros.eps)} · min_samples = ${numberFormat.format(data.parametros.min_samples)}`;
    $("#dbscanExplanation").textContent = data.explicacion;

    const comparison = $("#dbscanComparison");
    const rows = Array.isArray(data.comparacion) ? data.comparacion : [];
    comparison.replaceChildren();
    if (!rows.length) {
      const row = document.createElement("tr");
      const cell = createTextElement("td", "", "El backend no devolvió grupos para estos parámetros.");
      cell.colSpan = 10;
      row.append(cell);
      comparison.append(row);
    } else {
      rows.forEach((group) => {
        const row = document.createElement("tr");
        [
          group.etiqueta,
          formatNumber(group.pacientes, 0),
          formatNumber(group.edad_promedio),
          formatNumber(group.pulso_promedio),
          formatNumber(group.sistolica_promedio),
          formatNumber(group.diastolica_promedio),
          formatNumber(group.glucosa_promedio),
          formatNumber(group.ckmb_mediana),
          formatNumber(group.troponina_mediana),
          `${formatNumber(group.porcentaje_positivos)}%`,
        ].forEach((value) => row.append(createTextElement("td", "", value)));
        comparison.append(row);
      });
    }

    renderDbscanCharts(Array.isArray(data.graficos) ? data.graficos : []);
  }

  function renderDbscanCharts(charts) {
    const container = $("#dbscanCharts");
    container.replaceChildren();
    charts.forEach((chart) => {
      const card = createTextElement("article", "surface-card dbscan-chart-card");
      const heading = createTextElement("div", "card-heading");
      heading.append(
        createTextElement("div", "", ""),
        createTextElement("span", "chart-badge", "COMPARACIÓN"),
      );
      const headingText = createTextElement("div", "");
      headingText.append(
        createTextElement("h3", "", chart.titulo),
        createTextElement("p", "", chart.unidad),
      );
      heading.firstElementChild.replaceWith(headingText);
      card.append(heading);

      const values = Array.isArray(chart.valores) ? chart.valores : [];
      const max = Math.max(...values.map((entry) => Math.abs(Number(entry.valor) || 0)), 1);
      values.forEach((entry) => {
        const row = createTextElement("div", "dbscan-bar-row");
        const value = Number(entry.valor);
        const track = createTextElement("span", "importance-track");
        const fill = createTextElement("span", "dbscan-bar-fill");
        fill.style.width = `${Math.max(0, Math.min(100, Math.abs(value) / max * 100))}%`;
        track.append(fill);
        row.append(
          createTextElement("span", "importance-label", entry.etiqueta),
          track,
          createTextElement("strong", "importance-value", formatNumber(value)),
        );
        card.append(row);
      });
      container.append(card);
    });

    if (!charts.length) {
      const empty = createTextElement("div", "empty-state");
      empty.append(
        createTextElement("span", "", "◷"),
        createTextElement("b", "", "No hay comparaciones disponibles"),
        createTextElement("small", "", "El backend no devolvió estadísticas para estos parámetros."),
      );
      container.append(empty);
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
    } catch (error) {
      $("#classificationNotice").hidden = false;
      $("#classificationNotice").textContent =
        `No fue posible cargar las métricas del clasificador: ${error.message}`;
    }
  }

  function renderClassification(data) {
    if (data.accuracy !== undefined) $("#modelAccuracy").textContent = `${formatNumber(Number(data.accuracy) * 100, 2)}%`;
    if (data.precision !== undefined) $("#modelPrecision").textContent = `${formatNumber(Number(data.precision) * 100, 2)}%`;
    if (data.recall !== undefined) $("#modelRecall").textContent = `${formatNumber(Number(data.recall) * 100, 2)}%`;
    if (data.f1_score !== undefined) $("#modelF1").textContent = `${formatNumber(Number(data.f1_score) * 100, 2)}%`;
    if (data.test_samples !== undefined) $("#testSamples").textContent = numberFormat.format(data.test_samples);
    if (data.configuracion) {
      const config = data.configuracion;
      const fingerprint = String(config.dataset_sha256 || "").slice(0, 12);
      $("#modelConfiguration").textContent =
        `${config.algoritmo} v${config.version} · ${config.n_estimators} árboles · ` +
        `test_size ${config.test_size} · random_state ${config.random_state} · ` +
        `scikit-learn ${config.sklearn_version} · ` +
        `datos SHA-256 ${fingerprint}`;
    }
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
    const predictorCount = $("#predictorCount");
    if (predictorCount) predictorCount.textContent = numberFormat.format(usable.length);
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

  function createTextElement(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = String(text);
    return element;
  }

  function renderPredictionPlaceholder(container) {
    container.className = "surface-card result-card";
    const content = createTextElement("div", "result-placeholder");
    const illustration = createTextElement("div", "result-illustration");
    illustration.append(
      createTextElement("span", "result-ring"),
      createTextElement("span", "result-heart", "♡"),
      createTextElement("i", "result-spark spark-one", "✦"),
      createTextElement("i", "result-spark spark-two", "✦"),
    );
    content.append(
      illustration,
      createTextElement("span", "result-eyebrow", "RESULTADO DEL MODELO"),
      createTextElement("h3", "", "Tu resultado aparecerá aquí"),
      createTextElement("p", "", "Completa los datos y consulta el servicio de predicción."),
    );
    container.replaceChildren(content);
  }

  function renderClusterPlaceholder(container) {
    container.className = "surface-card cluster-result-card";
    const content = createTextElement("div", "cluster-result-placeholder");
    content.append(
      createTextElement("span", "result-eyebrow", "GRUPO DEL PACIENTE"),
      createTextElement("h3", "", "El grupo aparecerá aquí"),
      createTextElement("p", "", "Completa los datos para consultar su asignación a un grupo."),
    );
    container.replaceChildren(content);
  }

  function clearPredictionPanels() {
    renderPredictionPlaceholder($("#predictionResult"));
    renderClusterPlaceholder($("#clusterResult"));
    renderDecisionTreePlaceholder();
  }

  function renderPredictionLoading(container) {
    container.className = "surface-card result-card";
    const content = createTextElement("div", "result-placeholder");
    content.append(
      createTextElement("span", "loader"),
      createTextElement("p", "", "Consultando el modelo…"),
    );
    container.replaceChildren(content);
  }

  function renderClusterLoading(container) {
    container.className = "surface-card cluster-result-card";
    const content = createTextElement("div", "cluster-result-loading");
    content.append(
      createTextElement("span", "loader"),
      createTextElement("span", "", "Consultando el grupo…"),
    );
    container.replaceChildren(content);
  }

  function renderPredictionError(error, container) {
    container.className = "surface-card result-card result-error";
    const content = createTextElement("div", "result-placeholder");
    content.append(
      createTextElement("span", "result-outcome-icon", "!"),
      createTextElement("span", "result-outcome-label", "SERVICIO NO DISPONIBLE"),
      createTextElement("h3", "", "No se pudo consultar el modelo"),
      createTextElement("p", "", `No fue posible obtener la clasificación: ${error.message}`),
    );
    container.replaceChildren(content);
    renderDecisionTreeError(error);
  }

  function renderClusterError(error, container) {
    container.className = "surface-card cluster-result-card";
    const content = createTextElement("div", "cluster-result-error");
    content.append(
      createTextElement("span", "result-outcome-icon", "!"),
      createTextElement("span", "result-outcome-label", "GRUPO NO DISPONIBLE"),
      createTextElement("h3", "", "No se pudo asignar el grupo"),
      createTextElement("p", "", error.message),
    );
    container.replaceChildren(content);
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
    const content = createTextElement("div", "");

    content.append(
      createTextElement("span", "result-outcome-icon", positive ? "!" : "✓"),
      createTextElement("span", "result-outcome-label", "RESULTADO DEL MODELO"),
      createTextElement("h3", "", label),
      createTextElement("span", "result-class-label", `Clase estimada: ${data.clase ?? rawPrediction}`),
      createTextElement(
        "p",
        "",
        positive
          ? "La salida del modelo sugiere una clase positiva. Solicita valoración de un profesional de la salud."
          : "La salida del modelo corresponde a una clase negativa. Esto no descarta una condición clínica.",
      ),
    );

    if (probability !== undefined) {
      const percentage = Number(probability) <= 1 ? Number(probability) * 100 : probability;
      content.append(
        createTextElement("strong", "probability-value", `${formatNumber(percentage, 1)}%`),
        createTextElement("span", "probability-caption", "Confianza de la clase predicha"),
      );
    }

    const probabilities = data.probabilities ?? data.probabilidades;
    if (probabilities && typeof probabilities === "object") {
      const chart = createTextElement("div", "prediction-probabilities");
      Object.entries(probabilities).forEach(([className, rawValue]) => {
        const value = Number(rawValue);
        const percentage = value <= 1 ? value * 100 : value;
        const row = createTextElement("div", "probability-row");
        const track = createTextElement("span", "probability-track");
        const fill = createTextElement("span", "probability-fill");
        fill.style.width = `${Math.max(0, Math.min(100, percentage))}%`;
        track.append(fill);
        row.append(
          createTextElement("span", "probability-class", className),
          track,
          createTextElement("strong", "probability-percent", `${formatNumber(percentage, 1)}%`),
        );
        chart.append(row);
      });
      content.append(
        createTextElement("span", "probability-caption", "Estimaciones del modelo por clase"),
        chart,
        createTextElement(
          "p",
          "probability-disclaimer",
          "No son probabilidades clínicas calibradas ni constituyen un diagnóstico.",
        ),
      );
    }

    container.className = `surface-card result-card ${positive ? "result-positive" : "result-negative"}`;
    container.replaceChildren(content);
  }

  function renderDecisionTreePlaceholder(message = "Consulta un paciente para cargar la estructura real del árbol y resaltar su recorrido.") {
    activeTreeView = null;
    $("#decisionTreeMeta").textContent = "La visualización aparecerá después de consultar el modelo.";
    $("#decisionTreeSvg").setAttribute("hidden", "");
    $("#decisionTreePlaceholder").hidden = false;
    $("#decisionTreePlaceholder").textContent = message;
    $("#treeNodeDetail").textContent = "Selecciona un nodo del árbol para ver sus detalles.";
  }

  function renderDecisionTreeLoading() {
    renderDecisionTreePlaceholder("Cargando la estructura del árbol y el recorrido del paciente…");
  }

  function renderDecisionTreeError(error) {
    renderDecisionTreePlaceholder(`El árbol no está disponible: ${error.message}`);
  }

  function createSvgElement(tag, attributes = {}) {
    const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.entries(attributes).forEach(([name, value]) => element.setAttribute(name, String(value)));
    return element;
  }

  function renderTreeNodeDetail(node) {
    const detail = $("#treeNodeDetail");
    const heading = node.es_hoja ? `Hoja ${node.id}` : `Nodo ${node.id}`;
    detail.replaceChildren(
      createTextElement("strong", "", heading),
      createTextElement("span", "", `Muestras: ${numberFormat.format(node.muestras)}`),
      createTextElement("span", "", `Clase del nodo: ${node.clase}`),
      createTextElement(
        "p",
        "",
        node.condicion_evaluada || node.condicion || "Nodo sin condición de división.",
      ),
    );
  }

  function renderDecisionTree(tree) {
    if (!tree || !Array.isArray(tree.nodos) || !tree.nodos.length) {
      throw new Error("La respuesta no contiene la estructura del árbol.");
    }
    const svg = $("#decisionTreeSvg");
    const placeholder = $("#decisionTreePlaceholder");
    const nodeById = new Map(tree.nodos.map((node) => [Number(node.id), node]));
    const positions = new Map();
    let leafIndex = 0;
    let maxDepth = 0;

    function placeNode(nodeId, depth) {
      const node = nodeById.get(nodeId);
      if (!node) return { x: 80 + leafIndex++ * 150, y: 60 + depth * 105 };
      maxDepth = Math.max(maxDepth, depth);
      const hasChildren = node.izquierda !== null && node.derecha !== null;
      let x;
      if (!hasChildren) {
        x = 80 + leafIndex++ * 150;
      } else {
        const left = placeNode(Number(node.izquierda), depth + 1);
        const right = placeNode(Number(node.derecha), depth + 1);
        x = (left.x + right.x) / 2;
      }
      const point = { x, y: 60 + depth * 105 };
      positions.set(nodeId, point);
      return point;
    }

    placeNode(0, 0);
    const width = Math.max(900, leafIndex * 150 + 80);
    const height = Math.max(420, (maxDepth + 1) * 105 + 70);
    svg.replaceChildren();
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.setAttribute("preserveAspectRatio", "xMinYMin meet");

    tree.nodos.forEach((node) => {
      const parent = positions.get(Number(node.id));
      if (!parent || node.izquierda === null || node.derecha === null) return;
      [
        [Number(node.izquierda), "izquierda", "≤"],
        [Number(node.derecha), "derecha", ">"],
      ].forEach(([childId, branch, label]) => {
        const child = positions.get(childId);
        if (!child) return;
        const edge = createSvgElement("line", {
          x1: parent.x,
          y1: parent.y + 26,
          x2: child.x,
          y2: child.y - 26,
          class: node.visitado && nodeById.get(childId)?.visitado ? "tree-edge tree-edge-active" : "tree-edge",
        });
        svg.append(edge);
        const branchLabel = createSvgElement("text", {
          x: (parent.x + child.x) / 2,
          y: (parent.y + child.y) / 2,
          class: "tree-edge-label",
        });
        branchLabel.textContent = label;
        svg.append(branchLabel);
      });
    });

    tree.nodos.forEach((node) => {
      const point = positions.get(Number(node.id));
      if (!point) return;
      const group = createSvgElement("g", {
        transform: `translate(${point.x - 62}, ${point.y - 26})`,
        class: `tree-node${node.visitado ? " is-visited" : ""}${node.es_hoja ? " is-leaf" : ""}${node.clase === "positive" ? " is-positive" : ""}`,
        tabindex: "0",
        role: "button",
        "aria-label": `${node.es_hoja ? "Hoja" : "Nodo"} ${node.id}, ${node.variable || node.clase}, ${node.muestras} muestras`,
      });
      const title = createSvgElement("title");
      title.textContent = node.condicion_evaluada || node.condicion || `Clase ${node.clase}`;
      const rect = createSvgElement("rect", { width: 124, height: 52, rx: 7 });
      const firstLine = createSvgElement("text", { x: 62, y: 17, class: "tree-node-label" });
      firstLine.textContent = node.es_hoja ? `Hoja · ${node.clase}` : String(node.variable);
      const secondLine = createSvgElement("text", { x: 62, y: 32, class: "tree-node-condition" });
      secondLine.textContent = node.es_hoja ? `Nodo ${node.id}` : `≤ ${formatNumber(node.umbral, 3)}`;
      const thirdLine = createSvgElement("text", { x: 62, y: 45, class: "tree-node-samples" });
      thirdLine.textContent = `n = ${numberFormat.format(node.muestras)}`;
      group.append(title, rect, firstLine, secondLine, thirdLine);
      group.addEventListener("click", () => renderTreeNodeDetail(node));
      group.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          renderTreeNodeDetail(node);
        }
      });
      svg.append(group);
    });

    const pathNodes = tree.nodos.filter((node) => node.visitado);
    const pathPoints = pathNodes.map((node) => positions.get(Number(node.id))).filter(Boolean);
    const minX = Math.min(...pathPoints.map((point) => point.x));
    const maxX = Math.max(...pathPoints.map((point) => point.x));
    const focus = {
      x: Math.max(0, minX - 180),
      y: 0,
      width: Math.min(width, Math.max(900, maxX - minX + 360)),
      height,
    };
    focus.x = Math.min(focus.x, width - focus.width);
    activeTreeView = { svg, width, height, focus, current: { ...focus } };
    applyTreeView();
    let panStart = null;
    svg.addEventListener("pointerdown", (event) => {
      if (event.target.closest(".tree-node")) return;
      panStart = {
        pointerId: event.pointerId,
        x: event.clientX,
        y: event.clientY,
        view: { ...activeTreeView.current },
      };
      svg.setPointerCapture(event.pointerId);
    });
    svg.addEventListener("pointermove", (event) => {
      if (!panStart || event.pointerId !== panStart.pointerId) return;
      const bounds = svg.getBoundingClientRect();
      activeTreeView.current = {
        ...panStart.view,
        x: Math.max(0, Math.min(
          width - panStart.view.width,
          panStart.view.x - (event.clientX - panStart.x) * panStart.view.width / bounds.width,
        )),
        y: Math.max(0, Math.min(
          height - panStart.view.height,
          panStart.view.y - (event.clientY - panStart.y) * panStart.view.height / bounds.height,
        )),
      };
      applyTreeView();
    });
    svg.addEventListener("pointerup", () => { panStart = null; });
    svg.addEventListener("pointercancel", () => { panStart = null; });
    $("#decisionTreeMeta").textContent =
      `Árbol ${tree.indice_arbol} de ${tree.total_arboles} · profundidad ${tree.profundidad} · ${tree.total_nodos} nodos`;
    $("#treeNodeDetail").textContent =
      `Recorrido resaltado: ${pathNodes.length} nodos; hoja alcanzada ${tree.nodo_hoja}. Selecciona un nodo para ver sus detalles.`;
    placeholder.hidden = true;
    svg.removeAttribute("hidden");
  }

  function applyTreeView() {
    if (!activeTreeView) return;
    const { current, svg } = activeTreeView;
    svg.setAttribute("viewBox", `${current.x} ${current.y} ${current.width} ${current.height}`);
  }

  function zoomDecisionTree(action) {
    if (!activeTreeView) return;
    const view = activeTreeView.current;
    if (action === "reset") {
      activeTreeView.current = { ...activeTreeView.focus };
    } else {
      const factor = action === "in" ? 0.8 : 1.25;
      const width = Math.min(activeTreeView.width, Math.max(300, view.width * factor));
      const height = Math.min(activeTreeView.height, Math.max(240, view.height * factor));
      activeTreeView.current = {
        x: Math.max(0, Math.min(activeTreeView.width - width, view.x + (view.width - width) / 2)),
        y: Math.max(0, Math.min(activeTreeView.height - height, view.y + (view.height - height) / 2)),
        width,
        height,
      };
    }
    applyTreeView();
  }

  function renderCluster(data, container) {
    if (typeof data.nombre !== "string" || !data.perfil) {
      throw new Error("La respuesta de clustering no contiene el nombre y perfil del grupo.");
    }

    const content = createTextElement("div", "");
    const heading = createTextElement("div", "cluster-result-heading");
    const headingText = createTextElement("div", "");
    headingText.append(
      createTextElement("span", "result-eyebrow", "GRUPO DEL PACIENTE"),
      createTextElement("h3", "cluster-result-name", data.nombre),
    );
    heading.append(
      headingText,
      createTextElement("span", "cluster-result-id", `Grupo ${data.cluster_id}`),
    );

    const summary = createTextElement("div", "cluster-summary-grid");
    const patientCount = createTextElement("div", "cluster-summary-item");
    patientCount.append(
      createTextElement("span", "", "Pacientes en el grupo"),
      createTextElement("strong", "", formatNumber(data.pacientes_en_grupo, 0)),
    );
    const infarctRate = createTextElement("div", "cluster-summary-item");
    infarctRate.append(
      createTextElement("span", "", "Infartos en el grupo"),
      createTextElement("strong", "", `${formatNumber(data.pct_infarto_en_grupo, 2)}%`),
    );
    summary.append(patientCount, infarctRate);

    const profileTitle = createTextElement("h4", "cluster-profile-title", "Perfil del grupo");
    const profile = createTextElement("div", "cluster-profile-grid");
    [
      ["Edad media", data.perfil.edad],
      ["Pulso medio", data.perfil.pulso],
      ["Presión sistólica media", data.perfil.sistolica],
      ["Presión diastólica media", data.perfil.diastolica],
      ["Glucosa media", data.perfil.glucosa],
      ["Mediana CK-MB", data.perfil.ckmb_mediana],
      ["Mediana troponina", data.perfil.troponina_mediana],
    ].forEach(([label, value]) => {
      const item = createTextElement("div", "cluster-profile-item");
      item.append(
        createTextElement("span", "", label),
        createTextElement("strong", "", formatNumber(value)),
      );
      profile.append(item);
    });

    content.append(
      heading,
      summary,
      profileTitle,
      profile,
      createTextElement("p", "cluster-disclaimer", CLUSTER_DISCLAIMER),
    );
    container.className = "surface-card cluster-result-card";
    container.replaceChildren(content);
  }

  /* Pacientes reales del conjunto de datos (índice de la fila tras la limpieza). */
  const EXAMPLE_PATIENTS = {
    sano: {
      fila: 135,
      nota: "Paciente sano: troponina y CK-MB normales. El modelo lo clasifica negativo y lo ubica en un grupo de marcadores normales.",
      datos: { Age: 67, "Heart rate": 58, "Systolic blood pressure": 119, "Diastolic blood pressure": 72, "Blood sugar": 109, "CK-MB": 1.5, Troponin: 0.009 },
    },
    troponina: {
      fila: 25,
      nota: "Troponina de 1,39 ng/mL, muy por encima de lo normal. Es el marcador con más peso en el modelo: infarto con certeza casi total.",
      datos: { Age: 72, "Heart rate": 64, "Systolic blood pressure": 106, "Diastolic blood pressure": 68, "Blood sugar": 111, "CK-MB": 2.11, Troponin: 1.39 },
    },
    ckmb: {
      fila: 23,
      nota: "Paciente de 30 años con CK-MB de 50,5 ng/mL y troponina normal. La edad no engaña al modelo: detecta el infarto por la CK-MB.",
      datos: { Age: 30, "Heart rate": 63, "Systolic blood pressure": 110, "Diastolic blood pressure": 68, "Blood sugar": 107, "CK-MB": 50.46, Troponin: 0.003 },
    },
    limite: {
      fila: 35,
      nota: "Marcadores moderados. El clustering lo ubica en un grupo de marcadores normales, pero el clasificador sí detecta el infarto: lo supervisado complementa a lo no supervisado.",
      datos: { Age: 64, "Heart rate": 58, "Systolic blood pressure": 156, "Diastolic blood pressure": 76, "Blood sugar": 82, "CK-MB": 6.78, Troponin: 0.012 },
    },
    glucosa: {
      fila: 4,
      nota: "Glucosa de 300 mg/dL pero marcadores cardíacos bajos. El clustering lo agrupa por glucosa alta y el modelo, correctamente, no lo marca como infarto.",
      datos: { Age: 55, "Heart rate": 64, "Systolic blood pressure": 112, "Diastolic blood pressure": 65, "Blood sugar": 300, "CK-MB": 1.08, Troponin: 0.003 },
    },
  };

  function loadExamplePatient(key) {
    const example = EXAMPLE_PATIENTS[key];
    if (!example) return;
    const form = $("#predictionForm");
    Object.entries(example.datos).forEach(([name, value]) => {
      const input = form.elements.namedItem(name);
      if (input) input.value = value;
    });
    document.querySelectorAll(".example-button").forEach((button) => {
      button.classList.toggle("active", button.dataset.example === key);
    });
    const note = $("#exampleNote");
    note.textContent = `${example.nota} (Fila ${example.fila} del conjunto de datos.)`;
    note.hidden = false;
    form.requestSubmit();
  }

  async function submitPrediction(event) {
    event.preventDefault();
    const form = event.currentTarget;
    if (!form.reportValidity()) {
      clearPredictionPanels();
      return;
    }

    const button = $("#predictButton");
    const buttonLabel = $("#predictButtonLabel");
    const predictionResult = $("#predictionResult");
    const clusterResult = $("#clusterResult");
    const formData = new FormData(form);
    const patient = Object.fromEntries([...formData.entries()].map(([key, value]) => [key, Number(value)]));
    const requestOptions = {
      method: "POST",
      body: JSON.stringify(patient),
    };

    button.disabled = true;
    buttonLabel.textContent = "Consultando modelos…";
    renderPredictionLoading(predictionResult);
    renderClusterLoading(clusterResult);

    const predictionRequest = request(API.prediction, requestOptions)
      .then((data) => {
        renderPrediction(data, predictionResult);
        renderDecisionTree(data.explicacion_arbol);
      })
      .catch((error) => renderPredictionError(error, predictionResult));
    const clusterRequest = request(API.clusterAssignment, requestOptions)
      .then((data) => renderCluster(data, clusterResult))
      .catch((error) => renderClusterError(error, clusterResult));

    renderDecisionTreeLoading();
    try {
      await Promise.all([predictionRequest, clusterRequest]);
    } finally {
      button.disabled = false;
      buttonLabel.textContent = "Consultar modelo";
    }
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

  /* ===================== Ensamblado: Regresión, MDP y RL ===================== */
  const loadedSections = new Set();
  const actionPill = (action) => `<span class="action-pill action-${escapeHtml(action)}">${escapeHtml(action)}</span>`;

  function emptyRow(colspan, message) {
    return `<tr><td colspan="${colspan}"><div class="empty-state"><span>◷</span><b>${escapeHtml(message)}</b></div></td></tr>`;
  }

  function renderBars(container, rows, { valueDigits = 3, signed = false } = {}) {
    const max = Math.max(...rows.map((row) => Math.abs(Number(row.value))), 1e-9);
    container.innerHTML = rows.map((row) => {
      const value = Number(row.value);
      const width = Math.max(2, (Math.abs(value) / max) * 100);
      const negative = signed && value < 0;
      return `<div class="bar-row">
        <span class="bar-label" title="${escapeHtml(row.label)}">${escapeHtml(row.label)}</span>
        <span class="bar-track"><span class="bar-fill${negative ? " negative" : ""}" style="width:${width.toFixed(1)}%"></span></span>
        <span class="bar-value">${formatNumber(value, valueDigits)}</span>
      </div>`;
    }).join("");
  }

  async function loadSection(key, noticeSelector, loader) {
    if (loadedSections.has(key)) return;
    try {
      await loader();
      loadedSections.add(key);
      $(noticeSelector).hidden = true;
    } catch (error) {
      const notice = $(noticeSelector);
      notice.hidden = false;
      notice.innerHTML = `<span class="notice-symbol">!</span><span><strong>No fue posible cargar esta sección.</strong> ${escapeHtml(error.message)}</span>`;
    }
  }

  async function loadRegression() {
    const data = await request(API.regression);
    const best = data.resultados[0];
    $("#regBestModel").textContent = data.mejor_modelo;
    $("#regR2").textContent = formatNumber(best.r2, 3);
    $("#regMae").textContent = formatNumber(best.mae, 3);
    $("#regRmse").textContent = formatNumber(best.rmse, 3);
    $("#regTest").textContent = numberFormat.format(data.n_test);
    $("#regressionTable").innerHTML = data.resultados.map((row) => `<tr class="${row.modelo === data.mejor_modelo ? "best-row" : ""}">
      <td class="variable-cell">${escapeHtml(row.modelo)}</td>
      <td class="numeric-cell">${formatNumber(row.r2, 4)}</td>
      <td class="numeric-cell">${formatNumber(row.mae, 4)}</td>
      <td class="numeric-cell">${formatNumber(row.rmse, 4)}</td>
    </tr>`).join("");
    renderBars($("#regressionImportance"), data.importancias.map((item) => ({ label: item.variable, value: item.importancia })));
    const readings = {
      ESCALAR: "Riesgo alto: avisar al médico ya.",
      REPETIR: "Riesgo intermedio: repetir medición.",
      SEGUIMIENTO: "Riesgo bajo: control habitual.",
    };
    $("#regressionBridge").innerHTML = data.puente_mdp.map((row) => `<tr>
      <td class="numeric-cell">${formatNumber(row.riesgo, 2)}</td>
      <td>${actionPill(row.accion)}</td>
      <td>${escapeHtml(readings[row.accion] ?? "")}</td>
    </tr>`).join("");
  }

  async function loadMdp() {
    const data = await request(API.mdp);
    const actions = data.acciones;
    $("#mdpStatesCount").textContent = data.estados.length;
    $("#mdpActionsCount").textContent = actions.length;
    $("#mdpGamma").textContent = formatNumber(data.gamma, 2);
    const chosen = [...new Set(data.politica.map((row) => row.accion_optima))];
    $("#mdpPolicySummary").textContent = chosen.length === 1 ? chosen[0] : chosen.join(" / ");
    const agent = data.batalla.find((row) => /ptima/i.test(row.politica)) ?? data.batalla[0];
    $("#mdpBestValue").textContent = formatNumber(agent.v_promedio, 1);

    $("#mdpStates").innerHTML = data.estados.map((row) => `<tr>
      <td class="variable-cell">S${row.estado} · ${escapeHtml(row.nombre)}</td>
      <td class="numeric-cell">${numberFormat.format(row.pacientes)}</td>
      <td class="numeric-cell">${formatNumber(row.troponina_mediana, 3)}</td>
      <td class="numeric-cell">${formatNumber(row.ckmb_mediana, 2)}</td>
      <td class="numeric-cell">${formatNumber(row.pct_infarto, 1)} %</td>
    </tr>`).join("");

    $("#mdpReward").innerHTML = data.recompensa.map((row) => `<tr>
      <td class="variable-cell">${escapeHtml(row.estado)}</td>
      ${actions.map((action) => `<td class="numeric-cell">${formatNumber(row[action], 0)}</td>`).join("")}
    </tr>`).join("");

    $("#mdpPolicy").innerHTML = data.politica.map((row) => `<tr>
      <td class="variable-cell">${escapeHtml(row.estado)}</td>
      <td class="numeric-cell"><strong>${formatNumber(row.v_optimo, 2)}</strong></td>
      ${actions.map((action) => `<td class="numeric-cell">${formatNumber(row[`q_${action}`], 2)}</td>`).join("")}
      <td>${actionPill(row.accion_optima)}</td>
    </tr>`).join("");

    renderBars($("#mdpBattle"), data.batalla.map((row) => ({ label: row.politica, value: row.v_promedio })), { valueDigits: 1, signed: true });

    $("#mdpTransitions").innerHTML = actions.map((action) => {
      const block = data.transiciones[action];
      return `<div>
        <div class="matrix-title">${actionPill(action)}</div>
        <div class="table-wrap"><table class="matrix-mini">
          <thead><tr><th>s \\ s′</th>${block.estados.map((name) => `<th>${escapeHtml(name)}</th>`).join("")}</tr></thead>
          <tbody>${block.matriz.map((row, index) => `<tr><td class="variable-cell">${escapeHtml(block.estados[index])}</td>${row.map((value) => `<td class="${value >= 0.5 ? "cell-strong" : ""}">${formatNumber(value, 2)}</td>`).join("")}</tr>`).join("")}</tbody>
        </table></div>
      </div>`;
    }).join("");
  }

  async function loadRl() {
    const data = await request(API.rl);
    $("#rlTrain").textContent = numberFormat.format(data.n_entrenamiento);
    $("#rlTest").textContent = numberFormat.format(data.n_prueba);
    const best = data.comparacion.reduce((a, b) => (b.recompensa > a.recompensa ? b : a));
    const table = data.comparacion.find((row) => /Q-Table/i.test(row.politica) && /cuantil/i.test(row.politica))
      ?? data.comparacion.find((row) => /Q-Table/i.test(row.politica));
    $("#rlBestReward").textContent = formatNumber(best.recompensa, 2);
    $("#rlFn").textContent = numberFormat.format(best.falsos_negativos);
    $("#rlFa").textContent = numberFormat.format(best.falsas_alarmas);
    if (table) {
      $("#rlBestDelta").textContent = `vs ${formatNumber(table.recompensa, 2)} de la Q-Table`;
      $("#rlFaDelta").textContent = `vs ${numberFormat.format(table.falsas_alarmas)} de la Q-Table`;
    }

    $("#rlComparison").innerHTML = data.comparacion.map((row) => `<tr class="${row.politica === best.politica ? "best-row" : ""}">
      <td class="variable-cell">${escapeHtml(row.politica)}</td>
      <td class="numeric-cell">${formatNumber(row.recompensa, 2)}</td>
      <td class="numeric-cell">${numberFormat.format(row.falsos_negativos)}</td>
      <td class="numeric-cell">${numberFormat.format(row.falsas_alarmas)}</td>
      <td class="numeric-cell">${escapeHtml(row.sanos_en_verde)}</td>
      <td class="numeric-cell">${escapeHtml(row.pct_rojo)}</td>
      <td class="numeric-cell">${escapeHtml(row.pct_amarillo)}</td>
      <td class="numeric-cell">${escapeHtml(row.pct_verde)}</td>
    </tr>`).join("");

    const colors = ["Verde", "Amarillo", "Rojo"];
    $("#rlRewardMatrix").innerHTML = Object.entries(data.recompensa_triaje).map(([result, values]) => `<tr>
      <td class="variable-cell">${escapeHtml(result)}</td>
      ${colors.map((color) => `<td class="numeric-cell">${formatNumber(values[color], 1)}</td>`).join("")}
    </tr>`).join("");

    const experimentRows = (rows) => rows.map((row) => {
      const cells = Object.values(row);
      return `<tr><td class="variable-cell">${escapeHtml(cells[0])}</td><td class="numeric-cell">${escapeHtml(cells[1])}</td><td class="numeric-cell">${escapeHtml(cells[2])}</td></tr>`;
    }).join("");
    $("#rlStability").innerHTML = data.estabilidad.length ? experimentRows(data.estabilidad) : emptyRow(3, "Sin datos");
    $("#rlVariables").innerHTML = data.variables.length ? experimentRows(data.variables) : emptyRow(3, "Sin datos");
  }

  function switchTab(tabId) {
    if (!tabTitles[tabId]) return;
    document.querySelectorAll(".page-section").forEach((section) => section.classList.toggle("active", section.id === tabId));
    document.querySelectorAll(".nav-item").forEach((button) => button.classList.toggle("active", button.dataset.tab === tabId));
    $("#breadcrumbCurrent").textContent = tabTitles[tabId];
    history.replaceState(null, "", `#${tabId}`);
    $("#sidebar").classList.remove("open");
    $("#sidebarScrim").classList.remove("visible");
    if (tabId === "clustering") {
      loadClustering();
      loadDbscanAnalysis();
    }
    if (tabId === "classification") loadClassification();
    if (tabId === "regression") loadSection("regression", "#regressionNotice", loadRegression);
    if (tabId === "mdp") loadSection("mdp", "#mdpNotice", loadMdp);
    if (tabId === "rl") loadSection("rl", "#rlNotice", loadRl);
    window.scrollTo({ top: 0 });
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
    $("#predictionForm").addEventListener("invalid", clearPredictionPanels, true);
    $("#predictionForm").addEventListener("submit", submitPrediction);
    document.querySelectorAll("[data-example]").forEach((button) => {
      button.addEventListener("click", () => loadExamplePatient(button.dataset.example));
    });
    $("#dbscanForm").addEventListener("submit", (event) => {
      event.preventDefault();
      if (event.currentTarget.reportValidity()) loadDbscanAnalysis();
    });
    document.querySelectorAll("[data-tree-zoom]").forEach((button) => {
      button.addEventListener("click", () => zoomDecisionTree(button.dataset.treeZoom));
    });
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
