const BIT_OPTIONS = [1, 2, 4, 8];
const SAMPLE_IMAGE = "assets/ILSVRC2012_val_00004049.JPEG";

const imageInput = document.getElementById("imageInput");
const bitDepthInput = document.getElementById("bitDepth");
const bitLabel = document.getElementById("bitLabel");
const heroMode = document.getElementById("heroMode");
const heroBits = document.getElementById("heroBits");
const heroLevels = document.getElementById("heroLevels");
const heroSource = document.getElementById("heroSource");
const heroOriginalTotalBits = document.getElementById("heroOriginalTotalBits");
const heroQuantizedTotalBits = document.getElementById("heroQuantizedTotalBits");
const heroBitRatio = document.getElementById("heroBitRatio");
const heroOriginalDtype = document.getElementById("heroOriginalDtype");
const heroQuantizedDtype = document.getElementById("heroQuantizedDtype");
const activeBitTag = document.getElementById("activeBitTag");
const modeFormulaText = document.getElementById("modeFormulaText");
const downloadButton = document.getElementById("downloadButton");
const resetButton = document.getElementById("resetButton");
const modeInputs = [...document.querySelectorAll('input[name="colorMode"]')];

const originalCanvas = document.getElementById("originalCanvas");
const floatCanvas = document.getElementById("floatCanvas");
const quantizedCanvas = document.getElementById("quantizedCanvas");
const originalCtx = originalCanvas.getContext("2d", { willReadFrequently: true });
const floatCtx = floatCanvas.getContext("2d", { willReadFrequently: true });
const quantizedCtx = quantizedCanvas.getContext("2d", { willReadFrequently: true });
const galleryCanvases = [...document.querySelectorAll(".mini-canvas")];

const workingImage = new Image();
workingImage.crossOrigin = "anonymous";

let originalDisplayImageData = null;
let sourceFloatImage = null;
let currentSourceLabel = "Sample image";

function bitsFromSlider() {
  return BIT_OPTIONS[Number(bitDepthInput.value)];
}

function currentMode() {
  return modeInputs.find((input) => input.checked)?.value ?? "rgb";
}

function modeLabel(mode) {
  return mode === "gray" ? "Gray 1-channel" : "RGB";
}

function formatNumber(value) {
  return new Intl.NumberFormat("en-US").format(value);
}

function formatRatio(value) {
  if (!Number.isFinite(value)) {
    return "0x";
  }

  const rounded = value >= 10 ? Math.round(value) : Math.round(value * 10) / 10;
  return `${rounded}x`;
}

function imageDataTypeLabel(imageData) {
  return imageData?.data?.constructor?.name ?? "Unknown";
}

function cloneFloatImage(floatImage) {
  return {
    data: new Float32Array(floatImage.data),
    width: floatImage.width,
    height: floatImage.height,
  };
}

function imageDataToFloatImage(imageData) {
  return {
    data: Float32Array.from(imageData.data),
    width: imageData.width,
    height: imageData.height,
  };
}

function floatImageToImageData(floatImage) {
  const source = floatImage.data;
  const result = new Uint8ClampedArray(source.length);

  for (let index = 0; index < source.length; index += 1) {
    result[index] = Math.round(Math.min(255, Math.max(0, source[index])));
  }

  return new ImageData(result, floatImage.width, floatImage.height);
}

function setHeroMetrics(bits, mode) {
  const levels = 2 ** bits;
  const pixelCount = sourceFloatImage ? sourceFloatImage.width * sourceFloatImage.height : 0;
  const originalTotalBits = pixelCount * 3 * 32;
  const quantizedChannels = mode === "gray" ? 1 : 3;
  const quantizedTotalBits = pixelCount * quantizedChannels * bits;
  const bitRatio = quantizedTotalBits > 0 ? originalTotalBits / quantizedTotalBits : 0;
  const quantizedPreview = sourceFloatImage ? processFloatImage(sourceFloatImage, bits, mode) : null;

  bitLabel.textContent = `${bits}-bit`;
  heroMode.textContent = modeLabel(mode);
  heroBits.textContent = `${bits}-bit`;
  heroLevels.textContent = `${levels}`;
  activeBitTag.textContent = `${modeLabel(mode)} · ${bits}-bit`;
  heroSource.textContent = currentSourceLabel;
  heroOriginalTotalBits.textContent = formatNumber(originalTotalBits);
  heroQuantizedTotalBits.textContent = formatNumber(quantizedTotalBits);
  heroBitRatio.textContent = formatRatio(bitRatio);
  heroOriginalDtype.textContent = imageDataTypeLabel(sourceFloatImage);
  heroQuantizedDtype.textContent = imageDataTypeLabel(quantizedPreview);
  modeFormulaText.textContent = mode === "gray"
    ? "Gray mode first converts RGB into one luminance channel, then quantizes that single channel."
    : "RGB mode quantizes each R, G, and B channel separately.";
}

function fitCanvasToImage(canvas, image) {
  canvas.width = image.width;
  canvas.height = image.height;
}

function fitCanvasToData(canvas, imageData) {
  canvas.width = imageData.width;
  canvas.height = imageData.height;
}

function toGrayFloatImage(floatImage) {
  const source = floatImage.data;
  const result = new Float32Array(source.length);

  // Convert RGB into a single luminance value and mirror it across the display channels.
  for (let index = 0; index < source.length; index += 4) {
    const red = source[index];
    const green = source[index + 1];
    const blue = source[index + 2];
    const gray = Math.round((0.299 * red) + (0.587 * green) + (0.114 * blue));

    result[index] = gray;
    result[index + 1] = gray;
    result[index + 2] = gray;
    result[index + 3] = source[index + 3];
  }

  return {
    data: result,
    width: floatImage.width,
    height: floatImage.height,
  };
}

function quantizeRgbFloatImage(floatImage, bits) {
  if (bits === 8) {
    return cloneFloatImage(floatImage);
  }

  const levels = 2 ** bits;
  const maxIndex = levels - 1;
  const step = 255 / maxIndex;
  const source = floatImage.data;
  const result = new Float32Array(source.length);

  // Map each RGB float channel from 0-255 into a smaller set of discrete levels.
  for (let index = 0; index < source.length; index += 4) {
    for (let channel = 0; channel < 3; channel += 1) {
      const pixel = source[index + channel];
      const levelIndex = Math.round((pixel * maxIndex) / 255);
      result[index + channel] = Math.round(levelIndex * step);
    }
    result[index + 3] = source[index + 3];
  }

  return {
    data: result,
    width: floatImage.width,
    height: floatImage.height,
  };
}

function quantizeGrayFloatImage(floatImage, bits) {
  const grayFloatImage = toGrayFloatImage(floatImage);

  if (bits === 8) {
    return grayFloatImage;
  }

  const levels = 2 ** bits;
  const maxIndex = levels - 1;
  const step = 255 / maxIndex;
  const source = grayFloatImage.data;
  const result = new Float32Array(source.length);

  // Quantize one grayscale channel, then copy that one-channel result into RGB for display.
  for (let index = 0; index < source.length; index += 4) {
    const gray = source[index];
    const levelIndex = Math.round((gray * maxIndex) / 255);
    const quantizedGray = Math.round(levelIndex * step);

    result[index] = quantizedGray;
    result[index + 1] = quantizedGray;
    result[index + 2] = quantizedGray;
    result[index + 3] = source[index + 3];
  }

  return {
    data: result,
    width: floatImage.width,
    height: floatImage.height,
  };
}

function processFloatImage(floatImage, bits, mode) {
  return mode === "gray"
    ? quantizeGrayFloatImage(floatImage, bits)
    : quantizeRgbFloatImage(floatImage, bits);
}

function drawOriginal() {
  fitCanvasToData(originalCanvas, originalDisplayImageData);
  originalCtx.putImageData(originalDisplayImageData, 0, 0);
}

function drawFloatSource() {
  const floatImageData = floatImageToImageData(sourceFloatImage);
  fitCanvasToData(floatCanvas, floatImageData);
  floatCtx.putImageData(floatImageData, 0, 0);
}

function drawQuantized(bits, mode) {
  const quantizedFloat = processFloatImage(sourceFloatImage, bits, mode);
  const quantizedImageData = floatImageToImageData(quantizedFloat);
  fitCanvasToData(quantizedCanvas, quantizedImageData);
  quantizedCtx.putImageData(quantizedImageData, 0, 0);
}

function drawGallery(mode) {
  galleryCanvases.forEach((canvas) => {
    const bits = Number(canvas.dataset.bits);
    const context = canvas.getContext("2d", { willReadFrequently: true });
    const quantizedFloat = processFloatImage(sourceFloatImage, bits, mode);
    const quantizedImageData = floatImageToImageData(quantizedFloat);
    fitCanvasToData(canvas, quantizedImageData);
    context.putImageData(quantizedImageData, 0, 0);
  });
}

function updateDisplay() {
  if (!sourceFloatImage || !originalDisplayImageData) {
    return;
  }

  const bits = bitsFromSlider();
  const mode = currentMode();
  setHeroMetrics(bits, mode);
  drawOriginal();
  drawFloatSource();
  drawQuantized(bits, mode);
}

function refreshAllViews() {
  updateDisplay();
  drawGallery(currentMode());
}

function loadImageFromElement(image) {
  fitCanvasToImage(originalCanvas, image);
  originalCtx.clearRect(0, 0, originalCanvas.width, originalCanvas.height);
  originalCtx.drawImage(image, 0, 0);
  originalDisplayImageData = originalCtx.getImageData(0, 0, originalCanvas.width, originalCanvas.height);
  sourceFloatImage = imageDataToFloatImage(originalDisplayImageData);
  refreshAllViews();
}

function loadFromUrl(url, sourceLabel) {
  currentSourceLabel = sourceLabel;
  workingImage.onload = () => loadImageFromElement(workingImage);
  workingImage.src = url;
}

function handleFileSelection(event) {
  const [file] = event.target.files;
  if (!file) {
    return;
  }

  currentSourceLabel = file.name;
  const reader = new FileReader();
  reader.onload = () => {
    workingImage.onload = () => loadImageFromElement(workingImage);
    workingImage.src = reader.result;
  };
  reader.readAsDataURL(file);
}

function downloadCurrentImage() {
  const bits = bitsFromSlider();
  const mode = currentMode();
  const link = document.createElement("a");
  link.href = quantizedCanvas.toDataURL("image/png");
  link.download = `quantized-${mode}-${bits}bit.png`;
  link.click();
}

function resetToSample() {
  imageInput.value = "";
  bitDepthInput.value = "3";
  document.getElementById("modeRgb").checked = true;
  loadFromUrl(SAMPLE_IMAGE, "Sample image");
}

bitDepthInput.addEventListener("input", updateDisplay);
imageInput.addEventListener("change", handleFileSelection);
downloadButton.addEventListener("click", downloadCurrentImage);
resetButton.addEventListener("click", resetToSample);
modeInputs.forEach((input) => input.addEventListener("change", refreshAllViews));

loadFromUrl(SAMPLE_IMAGE, "Sample image");
