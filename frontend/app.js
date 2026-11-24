/* ============================================================
   Fruit Classifier — frontend logic
   ============================================================ */

/* API base URL — empty string means same origin as the page.
   The FastAPI server serves both the frontend and the API on :8000.
   A page opened directly from disk (file://) still points at the local server. */
const API_BASE_URL = window.location.protocol === "file:" ? "http://127.0.0.1:8000" : "";
const API_ORIGIN = API_BASE_URL || window.location.origin;

const MAX_FILE_BYTES = 10 * 1024 * 1024;
const ALLOWED_TYPES = ["image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp", "image/gif"];
const REQUEST_TIMEOUT_MS = 60000;

/* ---------- Element references ---------- */
const dropzone = document.getElementById("dropzone");
const dropzoneEmpty = document.getElementById("dropzoneEmpty");
const dropzonePreview = document.getElementById("dropzonePreview");
const fileInput = document.getElementById("fileInput");
const browseBtn = document.getElementById("browseBtn");
const openCameraBtn = document.getElementById("openCameraBtn");
const previewImage = document.getElementById("previewImage");
const removeBtn = document.getElementById("removeBtn");
const retakeBtn = document.getElementById("retakeBtn");
const cameraPanel = document.getElementById("cameraPanel");
const cameraVideo = document.getElementById("cameraVideo");
const captureBtn = document.getElementById("captureBtn");
const closeCameraBtn = document.getElementById("closeCameraBtn");
const fileMeta = document.getElementById("fileMeta");
const predictBtn = document.getElementById("predictBtn");
const predictLabel = predictBtn.querySelector(".btn-label");
const alertBox = document.getElementById("alert");

const resultPanel = document.getElementById("resultPanel");
const resultPlaceholder = document.getElementById("resultPlaceholder");
const resultContent = document.getElementById("resultContent");
const resultImage = document.getElementById("resultImage");
const resultKicker = document.getElementById("resultKicker");
const resultName = document.getElementById("resultName");
const resultNote = document.getElementById("resultNote");
const confidenceValue = document.getElementById("confidenceValue");
const confidenceBar = document.getElementById("confidenceBar");
const confidenceFill = document.getElementById("confidenceFill");
const probList = document.getElementById("probList");
const probCount = document.getElementById("probCount");
const resetResultBtn = document.getElementById("resetResultBtn");

const apiStatus = document.getElementById("apiStatus");
const apiLabel = apiStatus.querySelector(".api-label");

/* ---------- State ---------- */
let selectedFile = null;
let previewUrl = null;
let resultUrl = null;
let isBusy = false;
let apiOnlineLabel = "API online";
let cameraStream = null;
let cameraStarting = false;

/* ============================================================
   Helpers
   ============================================================ */
function showAlert(message, type = "error") {
  alertBox.textContent = message;
  alertBox.classList.toggle("alert--info", type === "info");
  alertBox.hidden = false;
}

function hideAlert() {
  alertBox.hidden = true;
  alertBox.textContent = "";
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function setApiStatus(state, label) {
  apiStatus.dataset.state = state;
  apiLabel.textContent = label;
  apiStatus.title =
    state === "online"
      ? `Backend connected at ${API_ORIGIN}`
      : state === "offline"
        ? `Backend unreachable at ${API_ORIGIN}`
        : "Checking backend…";
}

async function fetchWithTimeout(url, options = {}, timeoutMs = REQUEST_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

/* ============================================================
   Backend health
   ============================================================ */
async function checkApiHealth() {
  setApiStatus("checking", "Checking API…");
  if (await probeHealth()) return;

  // The backend may still be warming up (the checkpoint is large).
  setApiStatus("checking", "Reconnecting…");
  await new Promise((resolve) => setTimeout(resolve, 4000));
  if (await probeHealth()) return;

  setApiStatus("offline", "API offline");
}

async function probeHealth() {
  try {
    const response = await fetchWithTimeout(`${API_BASE_URL}/health`, {}, 8000);
    if (!response.ok) return false;
    const data = await response.json();
    if (!data.model_loaded) return false;
    const device = data.device ? data.device.toUpperCase() : "";
    apiOnlineLabel = device ? `API online · ${device}` : "API online";
    setApiStatus("online", apiOnlineLabel);
    return true;
  } catch {
    return false;
  }
}

/* ============================================================
   File selection / preview
   ============================================================ */
function validateFile(file) {
  if (!file) return "Please choose an image file.";
  if (!ALLOWED_TYPES.includes(file.type)) {
    return "Unsupported file type. Please upload a JPG, PNG, WEBP, BMP or GIF image.";
  }
  if (file.size === 0) return "That file is empty. Please choose another image.";
  if (file.size > MAX_FILE_BYTES) {
    return `Image is too large (${formatBytes(file.size)}). Maximum size is 10 MB.`;
  }
  return null;
}

function setFile(file, source = "upload") {
  const error = validateFile(file);
  if (error) {
    showAlert(error);
    return;
  }

  hideAlert();
  selectedFile = file;

  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);

  previewImage.src = previewUrl;
  previewImage.alt = `Selected image: ${file.name}`;
  dropzoneEmpty.hidden = true;
  cameraPanel.hidden = true;
  dropzonePreview.hidden = false;
  dropzone.dataset.state = "preview";

  fileMeta.textContent = `${file.name} · ${formatBytes(file.size)}`;
  fileMeta.hidden = false;

  retakeBtn.hidden = source !== "camera";

  predictBtn.disabled = false;
  predictLabel.textContent = "Predict fruit";
}

function clearFile() {
  selectedFile = null;
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }
  previewImage.removeAttribute("src");
  previewImage.alt = "Selected image preview";
  dropzoneEmpty.hidden = false;
  dropzonePreview.hidden = true;
  dropzone.dataset.state = "empty";
  retakeBtn.hidden = true;
  fileMeta.hidden = true;
  fileMeta.textContent = "";
  fileInput.value = "";
  predictBtn.disabled = true;
  predictLabel.textContent = "Predict fruit";
  hideAlert();
}

function openFileDialog() {
  if (isBusy || isCameraActive()) return;
  fileInput.click();
}

/* ============================================================
   Drag & drop
   ============================================================ */
dropzone.addEventListener("click", (event) => {
  if (dropzone.dataset.state !== "empty") return;
  if (event.target.closest("button")) return;
  openFileDialog();
});

browseBtn.addEventListener("click", (event) => {
  event.stopPropagation();
  openFileDialog();
});

["dragenter", "dragover"].forEach((type) => {
  dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    event.stopPropagation();
    if (!isBusy) dropzone.classList.add("is-dragging");
  });
});

["dragleave", "dragend"].forEach((type) => {
  dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    event.stopPropagation();
    if (!dropzone.contains(event.relatedTarget)) {
      dropzone.classList.remove("is-dragging");
    }
  });
});

dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  event.stopPropagation();
  dropzone.classList.remove("is-dragging");
  if (isBusy || isCameraActive()) return;

  const files = event.dataTransfer?.files;
  if (!files || files.length === 0) return;
  if (files.length > 1) {
    showAlert("Please drop a single image at a time.");
    return;
  }
  setFile(files[0]);
});

/* Stop the browser from navigating when an image is dropped elsewhere. */
["dragover", "drop"].forEach((type) => {
  window.addEventListener(type, (event) => {
    if (!dropzone.contains(event.target)) event.preventDefault();
  });
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (file) setFile(file);
});

removeBtn.addEventListener("click", (event) => {
  event.stopPropagation();
  if (isBusy) return;
  clearFile();
});

/* ============================================================
   Camera capture (getUserMedia + Canvas)
   ============================================================ */
let cameraSession = 0;

function isCameraActive() {
  return cameraStarting || cameraStream !== null || dropzone.dataset.state === "camera";
}

function cameraSupported() {
  return Boolean(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

function stopCameraStream() {
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
  }
  if (cameraVideo.srcObject) {
    cameraVideo.pause();
    cameraVideo.srcObject = null;
  }
}

function setCameraStarting(starting) {
  cameraStarting = starting;
  openCameraBtn.disabled = starting;
  retakeBtn.disabled = starting;
  openCameraBtn.classList.toggle("is-loading", starting);
  openCameraBtn.querySelector(".camera-label").textContent = starting
    ? "Opening camera…"
    : "Open Camera";
}

function restoreUploadState() {
  cameraPanel.hidden = true;
  stopCameraStream();

  if (selectedFile) {
    dropzoneEmpty.hidden = true;
    dropzonePreview.hidden = false;
    dropzone.dataset.state = "preview";
    predictBtn.disabled = false;
  } else {
    dropzoneEmpty.hidden = false;
    dropzonePreview.hidden = true;
    dropzone.dataset.state = "empty";
    predictBtn.disabled = true;
  }
}

function cameraErrorMessage(error) {
  const name = error && error.name;

  if (name === "NotAllowedError" || name === "SecurityError" || name === "PermissionDeniedError") {
    return "Camera permission was denied. Allow camera access in your browser settings, or upload an image instead.";
  }
  if (name === "NotFoundError" || name === "DevicesNotFoundError") {
    return "No camera was found on this device. Please upload an image instead.";
  }
  if (name === "NotReadableError" || name === "TrackStartError") {
    return "The camera is being used by another app. Close it and try again, or upload an image.";
  }
  if (name === "OverconstrainedError" || name === "ConstraintNotSatisfiedError") {
    return "This camera does not support the requested settings. Please upload an image instead.";
  }
  return "Could not access the camera. Check your browser permissions and try again, or upload an image.";
}

async function openCamera() {
  if (isBusy || cameraStarting || cameraStream) return;
  hideAlert();

  if (!window.isSecureContext) {
    showAlert(
      "Camera access needs a secure connection. Open the site at http://localhost:8000 (or HTTPS), then try again."
    );
    return;
  }

  if (!cameraSupported()) {
    showAlert(
      "Camera capture is not supported on this device or browser. Please upload an image instead."
    );
    return;
  }

  const session = ++cameraSession;
  setCameraStarting(true);

  try {
    // Rear camera is preferred on mobile; desktop falls back to the default camera.
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
    });

    if (session !== cameraSession) {
      stream.getTracks().forEach((track) => track.stop());
      return;
    }

    cameraStream = stream;
    cameraVideo.srcObject = stream;

    dropzoneEmpty.hidden = true;
    dropzonePreview.hidden = true;
    cameraPanel.hidden = false;
    dropzone.dataset.state = "camera";
    predictBtn.disabled = true;

    try {
      await cameraVideo.play();
    } catch {
      /* Autoplay can be blocked; the live stream still renders. */
    }

    captureBtn.disabled = false;
  } catch (error) {
    if (session !== cameraSession) return;
    cameraPanel.hidden = true;
    restoreUploadState();
    showAlert(cameraErrorMessage(error));
  } finally {
    if (session === cameraSession) setCameraStarting(false);
  }
}

function closeCamera() {
  cameraSession += 1;
  setCameraStarting(false);
  cameraPanel.hidden = true;
  stopCameraStream();
  restoreUploadState();
  hideAlert();
}

function capturePhoto() {
  if (!cameraStream || isBusy) return;

  const width = cameraVideo.videoWidth;
  const height = cameraVideo.videoHeight;
  if (!width || !height) {
    showAlert("The camera is still starting. Please wait a moment and try again.");
    return;
  }

  const context = document.createElement("canvas").getContext("2d");
  if (!context) {
    showAlert("Photo capture is not supported in this browser. Please upload an image instead.");
    return;
  }

  const canvas = context.canvas;
  canvas.width = width;
  canvas.height = height;
  context.drawImage(cameraVideo, 0, 0, width, height);

  captureBtn.disabled = true;
  const session = cameraSession;

  canvas.toBlob(
    (blob) => {
      if (session !== cameraSession) return; // camera was closed while encoding
      captureBtn.disabled = false;

      if (!blob) {
        showAlert("Could not capture the photo. Please try again.");
        return;
      }

      const file = new File([blob], `camera-photo-${Date.now()}.jpg`, {
        type: "image/jpeg",
      });

      cameraPanel.hidden = true;
      stopCameraStream();
      setFile(file, "camera");
      hideAlert();
    },
    "image/jpeg",
    0.92
  );
}

openCameraBtn.addEventListener("click", (event) => {
  event.stopPropagation();
  openCamera();
});

retakeBtn.addEventListener("click", (event) => {
  event.stopPropagation();
  openCamera();
});

captureBtn.addEventListener("click", (event) => {
  event.stopPropagation();
  capturePhoto();
});

closeCameraBtn.addEventListener("click", (event) => {
  event.stopPropagation();
  closeCamera();
});

/* Release the camera when the page is closed or hidden. */
window.addEventListener("pagehide", () => {
  cameraSession += 1;
  stopCameraStream();
});

window.addEventListener("beforeunload", () => {
  stopCameraStream();
});

document.addEventListener("visibilitychange", () => {
  if (document.hidden && (cameraStream || cameraStarting || dropzone.dataset.state === "camera")) {
    closeCamera();
  }
});

/* Restore the upload card if the page is reopened from the back/forward cache. */
window.addEventListener("pageshow", (event) => {
  if (event.persisted && dropzone.dataset.state === "camera") {
    restoreUploadState();
  }
});

/* ============================================================
   Prediction
   ============================================================ */
function setBusy(busy) {
  isBusy = busy;
  predictBtn.disabled = busy || !selectedFile;
  predictBtn.classList.toggle("is-loading", busy);
  predictLabel.textContent = busy ? "Analyzing image…" : "Predict fruit";
  dropzone.setAttribute("aria-busy", String(busy));
}

async function predict() {
  if (!selectedFile || isBusy) return;

  setBusy(true);
  hideAlert();

  const formData = new FormData();
  formData.append("file", selectedFile, selectedFile.name);

  try {
    const response = await fetchWithTimeout(`${API_BASE_URL}/predict`, {
      method: "POST",
      body: formData,
    });

    let data = null;
    try {
      data = await response.json();
    } catch {
      data = null;
    }

    if (!response.ok) {
      const detail =
        data && typeof data.detail === "string"
          ? data.detail
          : `Prediction failed (HTTP ${response.status}).`;
      throw new Error(detail);
    }

    if (!data || !data.predicted_class || typeof data.confidence !== "number") {
      throw new Error("The server returned an unexpected response.");
    }

    renderResult(data);
    setApiStatus("online", apiOnlineLabel);
  } catch (error) {
    if (error.name === "AbortError") {
      showAlert("The request timed out. Please try again.");
    } else if (error instanceof TypeError) {
      showAlert(
        "Could not reach the backend. Make sure the FastAPI server is running, then try again."
      );
      setApiStatus("offline", "API offline");
    } else {
      showAlert(error.message || "Something went wrong while predicting.");
    }
  } finally {
    setBusy(false);
  }
}

predictBtn.addEventListener("click", predict);

/* ============================================================
   Result rendering
   ============================================================ */
function renderResult(data) {
  const isNoFruit = data.predicted_class === "no fruit";
  const displayName = isNoFruit
    ? "No Fruit Detected"
    : data.display_name || titleCase(data.predicted_class);

  resultPanel.classList.toggle("is-no-fruit", isNoFruit);
  resultPlaceholder.hidden = true;
  resultContent.hidden = false;

  resultKicker.textContent = isNoFruit ? "Model prediction" : "Predicted fruit";
  resultName.textContent = displayName;
  resultNote.textContent = isNoFruit
    ? "The model did not detect any fruit in this image."
    : "Identified from your uploaded image.";

  if (resultUrl) URL.revokeObjectURL(resultUrl);
  resultUrl = URL.createObjectURL(selectedFile);
  resultImage.src = resultUrl;
  resultImage.alt = `Classified image: ${selectedFile.name}`;

  const percent = Math.min(100, Math.max(0, data.confidence * 100));
  confidenceValue.textContent = `${percent.toFixed(1)}%`;
  confidenceBar.setAttribute("aria-valuenow", percent.toFixed(1));
  confidenceFill.style.width = "0%";
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      confidenceFill.style.width = `${percent}%`;
    });
  });

  renderProbabilities(data.probabilities || {});

  if (window.matchMedia("(max-width: 960px)").matches) {
    resultPanel.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function titleCase(text) {
  return text
    .split(" ")
    .map((word) => (word ? word[0].toUpperCase() + word.slice(1) : word))
    .join(" ");
}

/* "apple fruit" -> "Apple", "no fruit" -> "No Fruit" */
function prettyClass(name) {
  if (name === "no fruit") return "No Fruit";
  return titleCase(name.replace(/\s+fruit$/i, ""));
}

function renderProbabilities(probabilities) {
  const entries = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
  probList.replaceChildren();
  probCount.textContent = `${entries.length} classes`;

  entries.forEach(([name, probability], index) => {
    const percent = Math.min(100, Math.max(0, probability * 100));
    const label = prettyClass(name);

    const row = document.createElement("li");
    row.className = `prob-row${index === 0 ? " is-top" : ""}`;
    row.title = `${label}: ${percent.toFixed(2)}%`;

    const nameEl = document.createElement("span");
    nameEl.className = "prob-name";
    nameEl.textContent = label;

    const track = document.createElement("span");
    track.className = "prob-track";
    const fill = document.createElement("span");
    fill.className = "prob-fill";
    track.append(fill);

    const value = document.createElement("span");
    value.className = "prob-value";
    value.textContent = percent >= 10 ? `${percent.toFixed(1)}%` : `${percent.toFixed(2)}%`;

    row.append(nameEl, track, value);
    probList.append(row);

    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        fill.style.width = `${percent}%`;
      });
    });
  });
}

function resetResult() {
  resultContent.hidden = true;
  resultPlaceholder.hidden = false;
  resultPanel.classList.remove("is-no-fruit");
  confidenceFill.style.width = "0%";
  confidenceBar.setAttribute("aria-valuenow", "0");
  probList.replaceChildren();
  if (resultUrl) {
    URL.revokeObjectURL(resultUrl);
    resultUrl = null;
  }
  resultImage.removeAttribute("src");
}

resetResultBtn.addEventListener("click", resetResult);

/* ============================================================
   Section navigation — clean URLs, no hash fragments
   ============================================================ */
const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

function scrollToSection(id, moveFocus = false) {
  const target = document.getElementById(id);
  if (!target) return;

  target.scrollIntoView({
    behavior: prefersReducedMotion.matches ? "auto" : "smooth",
    block: "start",
  });

  if (moveFocus) {
    target.setAttribute("tabindex", "-1");
    target.focus({ preventScroll: true });
  }
}

document.querySelectorAll("[data-scroll]").forEach((link) => {
  link.addEventListener("click", (event) => {
    // Let modified clicks (new tab / new window) behave normally.
    if (event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) {
      return;
    }

    const id = link.dataset.scroll;
    if (!document.getElementById(id)) return;

    event.preventDefault();
    scrollToSection(id, link.classList.contains("skip-link"));
  });
});

/* Remove legacy hashes such as #home or #about from the address bar. */
function cleanUrlHash() {
  if (!window.location.hash) return;
  try {
    history.replaceState(null, "", window.location.pathname + window.location.search);
  } catch {
    /* Some browsers restrict history updates on file:// — ignore. */
  }
}

/* Also clean hashes added later (e.g. from an old bookmark or manual entry). */
window.addEventListener("hashchange", cleanUrlHash);

/* ============================================================
   Init
   ============================================================ */
document.getElementById("year").textContent = String(new Date().getFullYear());
cleanUrlHash();
checkApiHealth();
