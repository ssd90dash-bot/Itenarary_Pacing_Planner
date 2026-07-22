// Cost Lab client: gather sweep params, stream results over SSE, render report.
(function () {
  const MAX_CAPS = 4;
  const form = document.getElementById("sweep-form");
  if (!form) return;

  const capBoxes = () => Array.from(document.querySelectorAll(".cap-box"));

  // Enforce the output-cap ceiling (matches the server-side chart series limit).
  capBoxes().forEach((box) =>
    box.addEventListener("change", () => {
      const checked = capBoxes().filter((b) => b.checked);
      if (checked.length > MAX_CAPS) box.checked = false;
      updateEstimate();
    })
  );
  form.querySelectorAll('input[name="temperature"]').forEach((b) =>
    b.addEventListener("change", updateEstimate)
  );

  function selectedRuns() {
    const temps = form.querySelectorAll('input[name="temperature"]:checked').length;
    const caps = capBoxes().filter((b) => b.checked).length;
    const samples = parseInt(form.querySelector('input[name="samples"]').value, 10);
    return temps * caps * samples;
  }

  window.updateEstimate = function () {
    const runs = selectedRuns();
    const repair = form.querySelector('input[name="repair"]').checked;
    let cost = runs * (window.PER_RUN_COST || 0);
    if (repair) cost *= 1.5;
    const costStr = cost < 0.01 ? "$" + cost.toFixed(5) : "$" + cost.toFixed(4);
    document.getElementById("run-count").textContent = runs;
    document.getElementById("est-cost").textContent = costStr;
    document.getElementById("confirm-runs").textContent = runs;
    document.getElementById("confirm-cost").textContent = costStr;
  };
  updateEstimate();

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const params = new URLSearchParams();
    form.querySelectorAll('input[name="temperature"]:checked').forEach((b) => params.append("temperature", b.value));
    capBoxes().filter((b) => b.checked).forEach((b) => params.append("cap", b.value));
    params.append("samples", form.querySelector('input[name="samples"]').value);
    if (form.querySelector('input[name="repair"]').checked) params.append("repair", "1");
    startSweep(params);
  });

  function startSweep(params) {
    document.getElementById("start-btn").disabled = true;
    document.getElementById("progress").classList.remove("hidden");
    document.getElementById("report").classList.add("hidden");
    const rows = document.getElementById("run-rows");
    rows.innerHTML = "";

    const source = new EventSource("/lab/stream?" + params.toString());

    source.addEventListener("run", (ev) => {
      const d = JSON.parse(ev.data);
      const pct = Math.round((d.index / d.total) * 100);
      document.getElementById("progress-bar").style.width = pct + "%";
      document.getElementById("progress-label").textContent =
        `Run ${d.index}/${d.total} — T=${d.temperature}, cap=${d.cap} — ${d.status}`;
      const statusColor =
        d.status === "ok" ? "text-emerald-600" : d.status === "truncated" ? "text-amber-600" : "text-rose-600";
      const tr = document.createElement("tr");
      tr.className = "border-b border-slate-100 dark:border-slate-800";
      tr.innerHTML =
        `<td class="py-1">${d.temperature}</td><td>${d.cap}</td>` +
        `<td class="${statusColor}">${d.status}</td>` +
        `<td>${d.input.toLocaleString()}</td><td>${d.output.toLocaleString()}</td>` +
        `<td>${d.cost}</td><td>${d.violations === null ? "—" : d.violations}</td>`;
      rows.appendChild(tr);
    });

    source.addEventListener("done", (ev) => {
      source.close();
      document.getElementById("progress").querySelector("h3").textContent = "Complete";
      renderReport(JSON.parse(ev.data));
    });

    source.addEventListener("error", (ev) => {
      source.close();
      const msg = ev.data ? JSON.parse(ev.data).message : "The sweep connection dropped.";
      const report = document.getElementById("report");
      report.classList.remove("hidden");
      report.innerHTML =
        `<div class="rounded-md bg-rose-50 border border-rose-200 text-rose-700 px-4 py-3 text-sm dark:bg-rose-950/40 dark:border-rose-900 dark:text-rose-200">${msg}</div>`;
      document.getElementById("start-btn").disabled = false;
    });
  }

  function renderReport(d) {
    const report = document.getElementById("report");
    report.classList.remove("hidden");
    let html = "";

    if (d.recommendation) {
      html +=
        `<div class="rounded-md bg-emerald-50 border border-emerald-200 text-emerald-800 px-4 py-3 text-sm dark:bg-emerald-950/40 dark:border-emerald-900 dark:text-emerald-200">
          <p class="font-medium">Recommended: temperature ${d.recommendation.temperature}, cap ${d.recommendation.cap}.</p>
          <p class="mt-1">${d.recommendation.reason}</p>
          <pre class="mt-2 bg-black/10 dark:bg-black/30 rounded p-2 text-xs overflow-x-auto">${d.recommendation.env}</pre>
        </div>`;
    } else {
      html += `<div class="rounded-md bg-amber-50 border border-amber-200 text-amber-800 px-4 py-3 text-sm">${d.recommendation_reason}</div>`;
    }

    if (d.min_samples < 2) {
      html +=
        `<div class="rounded-md bg-amber-50 border border-amber-200 text-amber-800 px-4 py-3 text-sm dark:bg-amber-950/40 dark:border-amber-900 dark:text-amber-200">
          <strong>Indicative, not conclusive.</strong> One sample per cell — the same parameters give a different itinerary each run. Raise samples to 2–3 before acting on a small difference.
        </div>`;
    }

    html +=
      `<div class="grid grid-cols-2 sm:grid-cols-4 gap-3">
        ${statTile("Total spend", d.total_cost)}
        ${statTile("Total tokens", d.total_tokens.toLocaleString())}
        ${statTile("Wasted tokens", d.wasted_tokens.toLocaleString())}
        ${statTile("Samples/cell", d.min_samples)}
      </div>`;

    html +=
      `<div class="grid md:grid-cols-2 gap-4">
        ${chartCard("Cost per run vs temperature", d.charts.cost_temp)}
        ${chartCard("Cost vs quality (bottom-left is best)", d.charts.cost_viol)}
      </div>`;

    report.innerHTML = html;
  }

  function statTile(label, val) {
    return `<div class="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-4 text-center"><div class="text-2xl font-semibold">${val}</div><div class="text-xs text-slate-500">${label}</div></div>`;
  }
  function chartCard(title, svg) {
    if (!svg) return "";
    return `<div class="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-5"><h3 class="font-semibold mb-3">${title}</h3><div class="overflow-x-auto">${svg}</div></div>`;
  }
})();
