const form = document.querySelector("#symptom-form");
const textarea = document.querySelector("#symptoms");
const result = document.querySelector("#result");
const statusBadge = document.querySelector("#api-status");
const submitButton = document.querySelector("#submit-button");
const clearButton = document.querySelector("#clear-button");

function parseSymptoms(value) {
  return value
    .split(/[\n,]+/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function renderResult(data) {
  const symptoms = data.input_symptoms || [];
  result.className = "";
  result.innerHTML = `
    <h2 class="result-title">${data.predicted_disease || "Consult a doctor"}</h2>
    <p>Submitted symptoms:</p>
    <ul class="symptom-list">
      ${symptoms.map((symptom) => `<li>${escapeHtml(symptom)}</li>`).join("")}
    </ul>
  `;
}

function renderError(message) {
  result.className = "";
  result.innerHTML = `<p class="error">${escapeHtml(message)}</p>`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    if (!response.ok) throw new Error("Health check failed");
    statusBadge.textContent = "API Online";
    statusBadge.className = "status online";
  } catch {
    statusBadge.textContent = "API Offline";
    statusBadge.className = "status offline";
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const symptoms = parseSymptoms(textarea.value);

  if (symptoms.length === 0) {
    renderError("Please enter at least one symptom.");
    return;
  }

  submitButton.disabled = true;
  submitButton.textContent = "Checking";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symptoms }),
    });

    if (!response.ok) {
      throw new Error("The API could not process this request.");
    }

    renderResult(await response.json());
  } catch (error) {
    renderError(error.message || "Something went wrong.");
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "Check Symptoms";
  }
});

clearButton.addEventListener("click", () => {
  textarea.value = "";
  result.className = "empty-state";
  result.textContent = "Your result will appear here after you submit symptoms.";
  textarea.focus();
});

checkHealth();
