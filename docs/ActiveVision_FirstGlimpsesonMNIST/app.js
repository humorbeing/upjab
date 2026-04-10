"use strict";

const IMAGE_SIZE = 28;
const SAMPLE_PATH = "samples/mnist_samples.json";

const state = {
  samples: [],
  currentSample: null,
  image: makeBlackImage(IMAGE_SIZE, IMAGE_SIZE),
  memoryImage: makeBlackImage(IMAGE_SIZE, IMAGE_SIZE),
  cx: IMAGE_SIZE / 2,
  cy: IMAGE_SIZE / 2,
  cropSize: IMAGE_SIZE / 2,
  outputSize: 4,
  rememberCount: 0,
  accumulatedPixels: 0,
  blackout: [false, false, false, false, false],
  activePanel: null,
  displays: new Map(),
};

const elements = {
  canvases: {
    original: document.getElementById("originalCanvas"),
    cropped: document.getElementById("croppedCanvas"),
    resized: document.getElementById("resizedCanvas"),
    reconstructed: document.getElementById("reconstructedCanvas"),
    remembered: document.getElementById("rememberedCanvas"),
  },
  titles: {
    original: document.getElementById("originalTitle"),
    cropped: document.getElementById("croppedTitle"),
    resized: document.getElementById("resizedTitle"),
    reconstructed: document.getElementById("reconstructedTitle"),
    remembered: document.getElementById("rememberedTitle"),
  },
  changeImageButton: document.getElementById("changeImageButton"),
  rememberButton: document.getElementById("rememberButton"),
  resetButton: document.getElementById("resetButton"),
  sizeStatus: document.getElementById("sizeStatus"),
  rememberStatus: document.getElementById("rememberStatus"),
  pixelStatus: document.getElementById("pixelStatus"),
  sliders: {
    centerX: document.getElementById("centerXSlider"),
    centerY: document.getElementById("centerYSlider"),
    cropSize: document.getElementById("cropSizeSlider"),
    outputSize: document.getElementById("outputSizeSlider"),
  },
  values: {
    centerX: document.getElementById("centerXValue"),
    centerY: document.getElementById("centerYValue"),
    cropSize: document.getElementById("cropSizeValue"),
    outputSize: document.getElementById("outputSizeValue"),
  },
  blackouts: [...Array(5)].map((_, index) => document.getElementById(`blackout${index}`)),
};

function clamp(value, minimum, maximum) {
  return Math.min(Math.max(value, minimum), maximum);
}

function makeBlackImage(width, height) {
  return Array.from({ length: height }, () => Array(width).fill(0));
}

function cloneImage(image) {
  return image.map((row) => row.slice());
}

function getImageShape(image) {
  return { height: image.length, width: image[0]?.length ?? 0 };
}

function extractSquarePatch(image, cx, cy, size) {
  const { height, width } = getImageShape(image);
  const xStart = Math.floor(cx - size / 2);
  const yStart = Math.floor(cy - size / 2);
  const xEnd = xStart + size;
  const yEnd = yStart + size;
  const patch = makeBlackImage(size, size);

  const srcXStart = Math.max(0, xStart);
  const srcYStart = Math.max(0, yStart);
  const srcXEnd = Math.min(width, xEnd);
  const srcYEnd = Math.min(height, yEnd);

  if (srcXStart < srcXEnd && srcYStart < srcYEnd) {
    const destXStart = srcXStart - xStart;
    const destYStart = srcYStart - yStart;
    for (let y = srcYStart; y < srcYEnd; y += 1) {
      for (let x = srcXStart; x < srcXEnd; x += 1) {
        patch[destYStart + y - srcYStart][destXStart + x - srcXStart] = image[y][x];
      }
    }
  }

  return patch;
}

function resizePatch(patch, outSize) {
  const { height: srcHeight, width: srcWidth } = getImageShape(patch);

  if (outSize === 1) {
    let total = 0;
    for (const row of patch) {
      for (const value of row) total += value;
    }
    return [[total / Math.max(1, srcHeight * srcWidth)]];
  }

  const resized = makeBlackImage(outSize, outSize);
  const scale = (srcHeight - 1) / (outSize - 1);

  for (let r = 0; r < outSize; r += 1) {
    for (let c = 0; c < outSize; c += 1) {
      const srcR = r * scale;
      const srcC = c * scale;
      const r0 = Math.floor(srcR);
      const c0 = Math.floor(srcC);
      const r1 = Math.min(r0 + 1, srcHeight - 1);
      const c1 = Math.min(c0 + 1, srcWidth - 1);
      const dr = srcR - r0;
      const dc = srcC - c0;

      resized[r][c] =
        (1 - dr) * (1 - dc) * patch[r0][c0] +
        dr * (1 - dc) * patch[r1][c0] +
        (1 - dr) * dc * patch[r0][c1] +
        dr * dc * patch[r1][c1];
    }
  }

  return resized;
}

function nearestResize(patch, outSize) {
  const { height, width } = getImageShape(patch);
  const enlarged = makeBlackImage(outSize, outSize);

  for (let y = 0; y < outSize; y += 1) {
    for (let x = 0; x < outSize; x += 1) {
      const srcY = Math.min(height - 1, Math.floor((y * height) / outSize));
      const srcX = Math.min(width - 1, Math.floor((x * width) / outSize));
      enlarged[y][x] = patch[srcY][srcX];
    }
  }

  return enlarged;
}

function getCropBounds(image, cx, cy, size) {
  const { height, width } = getImageShape(image);
  const xStart = Math.floor(cx - size / 2);
  const yStart = Math.floor(cy - size / 2);
  const xEnd = xStart + size;
  const yEnd = yStart + size;

  const srcXStart = Math.max(0, xStart);
  const srcYStart = Math.max(0, yStart);
  const srcXEnd = Math.min(width, xEnd);
  const srcYEnd = Math.min(height, yEnd);

  const patchXStart = srcXStart - xStart;
  const patchYStart = srcYStart - yStart;
  const patchXEnd = patchXStart + Math.max(0, srcXEnd - srcXStart);
  const patchYEnd = patchYStart + Math.max(0, srcYEnd - srcYStart);

  return {
    srcXStart,
    srcYStart,
    srcXEnd,
    srcYEnd,
    patchXStart,
    patchYStart,
    patchXEnd,
    patchYEnd,
  };
}

function buildReconstructedView(image, cx, cy, size, resizedPatch) {
  const { height, width } = getImageShape(image);
  const output = makeBlackImage(width, height);
  const enlarged = nearestResize(resizedPatch, size);
  const bounds = getCropBounds(image, cx, cy, size);

  if (bounds.srcXStart < bounds.srcXEnd && bounds.srcYStart < bounds.srcYEnd) {
    for (let y = bounds.srcYStart; y < bounds.srcYEnd; y += 1) {
      for (let x = bounds.srcXStart; x < bounds.srcXEnd; x += 1) {
        output[y][x] = enlarged[bounds.patchYStart + y - bounds.srcYStart][bounds.patchXStart + x - bounds.srcXStart];
      }
    }
  }

  return output;
}

function copyCropIntoMemory(reconstructed) {
  const bounds = getCropBounds(state.image, state.cx, state.cy, state.cropSize);
  if (bounds.srcXStart >= bounds.srcXEnd || bounds.srcYStart >= bounds.srcYEnd) return;

  for (let y = bounds.srcYStart; y < bounds.srcYEnd; y += 1) {
    for (let x = bounds.srcXStart; x < bounds.srcXEnd; x += 1) {
      state.memoryImage[y][x] = reconstructed[y][x];
    }
  }
}

function prepareRenderData() {
  const patch = extractSquarePatch(state.image, state.cx, state.cy, state.cropSize);
  const resized = resizePatch(patch, state.outputSize);
  const reconstructed = buildReconstructedView(state.image, state.cx, state.cy, state.cropSize, resized);
  return { patch, resized, reconstructed };
}

function imageToCanvasBitmap(image) {
  const { height, width } = getImageShape(image);
  const bitmap = document.createElement("canvas");
  bitmap.width = Math.max(1, width);
  bitmap.height = Math.max(1, height);
  const ctx = bitmap.getContext("2d");
  const imageData = ctx.createImageData(bitmap.width, bitmap.height);

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const value = Math.round(clamp(image[y][x], 0, 1) * 255);
      const offset = (y * width + x) * 4;
      imageData.data[offset] = value;
      imageData.data[offset + 1] = value;
      imageData.data[offset + 2] = value;
      imageData.data[offset + 3] = 255;
    }
  }

  ctx.putImageData(imageData, 0, 0);
  return bitmap;
}

function resizeCanvasForDisplay(canvas) {
  const rect = canvas.getBoundingClientRect();
  const cssWidth = Math.max(1, Math.floor(rect.width));
  const cssHeight = Math.max(1, Math.floor(rect.height));
  const dpr = window.devicePixelRatio || 1;
  const targetWidth = Math.max(1, Math.floor(cssWidth * dpr));
  const targetHeight = Math.max(1, Math.floor(cssHeight * dpr));

  if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
    canvas.width = targetWidth;
    canvas.height = targetHeight;
  }

  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, cssWidth, cssHeight };
}

function drawPanel(canvas, image, options = {}) {
  const { ctx, cssWidth, cssHeight } = resizeCanvasForDisplay(canvas);
  const { height, width } = getImageShape(image);
  ctx.clearRect(0, 0, cssWidth, cssHeight);
  ctx.fillStyle = "#202422";
  ctx.fillRect(0, 0, cssWidth, cssHeight);

  const scale = Math.min(cssWidth / width, cssHeight / height);
  const displayWidth = Math.max(1, Math.floor(width * scale));
  const displayHeight = Math.max(1, Math.floor(height * scale));
  const offsetX = Math.floor((cssWidth - displayWidth) / 2);
  const offsetY = Math.floor((cssHeight - displayHeight) / 2);

  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(imageToCanvasBitmap(image), offsetX, offsetY, displayWidth, displayHeight);

  state.displays.set(canvas, {
    image,
    imageWidth: width,
    imageHeight: height,
    scale,
    offsetX,
    offsetY,
    displayWidth,
    displayHeight,
  });

  if (options.rect) {
    drawCropOverlay(ctx, options.rect, scale, offsetX, offsetY, options.overlayText);
  }
}

function drawCropOverlay(ctx, rect, scale, offsetX, offsetY, overlayText) {
  const { cx, cy, size } = rect;
  const left = offsetX + (cx - size / 2) * scale;
  const top = offsetY + (cy - size / 2) * scale;
  const width = size * scale;
  const height = size * scale;

  ctx.save();
  ctx.strokeStyle = "#ff4d4d";
  ctx.lineWidth = 2;
  ctx.strokeRect(left, top, width, height);

  if (overlayText) {
    const textX = left + width / 2;
    const textY = Math.max(offsetY + 16, top - 10);
    ctx.font = "bold 13px Arial, Helvetica, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.lineWidth = 4;
    ctx.strokeStyle = "#111614";
    ctx.fillStyle = "#32cd32";
    ctx.strokeText(overlayText, textX, textY);
    ctx.fillText(overlayText, textX, textY);
  }
  ctx.restore();
}

function canvasToImageCoords(canvas, clientX, clientY) {
  const display = state.displays.get(canvas);
  if (!display) return null;

  const rect = canvas.getBoundingClientRect();
  const localX = clientX - rect.left - display.offsetX;
  const localY = clientY - rect.top - display.offsetY;
  if (localX < 0 || localY < 0 || localX > display.displayWidth || localY > display.displayHeight) {
    return null;
  }

  return {
    x: clamp(localX / display.scale, 0, display.imageWidth),
    y: clamp(localY / display.scale, 0, display.imageHeight),
  };
}

function formatCenter(value) {
  return Number(value).toFixed(1);
}

function syncControls() {
  const { height, width } = getImageShape(state.image);
  const maxSquare = Math.max(1, Math.max(height, width));

  elements.sliders.centerX.max = String(width);
  elements.sliders.centerY.max = String(height);
  elements.sliders.cropSize.max = String(maxSquare);
  elements.sliders.outputSize.max = String(maxSquare);

  elements.sliders.centerX.value = String(state.cx);
  elements.sliders.centerY.value = String(state.cy);
  elements.sliders.cropSize.value = String(state.cropSize);
  elements.sliders.outputSize.value = String(state.outputSize);

  elements.values.centerX.value = formatCenter(state.cx);
  elements.values.centerY.value = formatCenter(state.cy);
  elements.values.cropSize.value = String(state.cropSize);
  elements.values.outputSize.value = String(state.outputSize);
}

function renderAll() {
  const { patch, resized, reconstructed } = prepareRenderData();
  const { height, width } = getImageShape(state.image);
  const sizeNote = `[${state.cropSize} x ${state.cropSize}] ==> [${state.outputSize} x ${state.outputSize}]`;
  const panelImages = [
    state.blackout[0] ? makeBlackImage(width, height) : state.image,
    state.blackout[1] ? makeBlackImage(state.cropSize, state.cropSize) : patch,
    state.blackout[2] ? makeBlackImage(state.outputSize, state.outputSize) : resized,
    state.blackout[3] ? makeBlackImage(width, height) : reconstructed,
    state.blackout[4] ? makeBlackImage(width, height) : state.memoryImage,
  ];

  drawPanel(elements.canvases.original, panelImages[0], {
    rect: { cx: state.cx, cy: state.cy, size: state.cropSize },
    overlayText: sizeNote,
  });
  drawPanel(elements.canvases.cropped, panelImages[1]);
  drawPanel(elements.canvases.resized, panelImages[2]);
  drawPanel(elements.canvases.reconstructed, panelImages[3], {
    rect: { cx: state.cx, cy: state.cy, size: state.cropSize },
    overlayText: sizeNote,
  });
  drawPanel(elements.canvases.remembered, panelImages[4], {
    rect: { cx: state.cx, cy: state.cy, size: state.cropSize },
    overlayText: sizeNote,
  });

  elements.titles.original.textContent = `Original (${width}x${height})`;
  elements.titles.cropped.textContent = `Cropped ${state.cropSize}x${state.cropSize}`;
  elements.titles.resized.textContent = `Resized ${state.outputSize}x${state.outputSize}`;
  elements.titles.reconstructed.textContent = `Reconstructed Crop (${width}x${height})`;
  elements.titles.remembered.textContent = `Remembered (${width}x${height})`;
  elements.sizeStatus.textContent = `Crop Size to N Size: ${sizeNote}`;
  elements.rememberStatus.textContent = `Remember Count: ${state.rememberCount}`;
  elements.pixelStatus.textContent = `Accumulated Pixels: ${state.accumulatedPixels}`;
  syncControls();
}

function rememberCurrentCrop() {
  const { resized, reconstructed } = prepareRenderData();
  copyCropIntoMemory(reconstructed);
  state.rememberCount += 1;
  state.accumulatedPixels += resized.length * resized.length;
  renderAll();
}

function resetMemory() {
  const { height, width } = getImageShape(state.image);
  state.memoryImage = makeBlackImage(width, height);
  state.rememberCount = 0;
  state.accumulatedPixels = 0;
  renderAll();
}

function setImage(sample, resetControls) {
  state.currentSample = sample;
  state.image = cloneImage(sample.pixels);
  const { height, width } = getImageShape(state.image);
  const maxSquare = Math.max(1, Math.max(height, width));
  state.memoryImage = makeBlackImage(width, height);
  state.rememberCount = 0;
  state.accumulatedPixels = 0;

  if (resetControls) {
    state.cx = width / 2;
    state.cy = height / 2;
    state.cropSize = Math.min(maxSquare, Math.max(1, Math.floor(maxSquare / 2)));
    state.outputSize = Math.min(maxSquare, 4);
  } else {
    state.cx = clamp(state.cx, 0, width);
    state.cy = clamp(state.cy, 0, height);
    state.cropSize = Math.round(clamp(state.cropSize, 1, maxSquare));
    state.outputSize = Math.round(clamp(state.outputSize, 1, maxSquare));
  }

  renderAll();
}

function changeImage() {
  if (!state.samples.length) return;
  const next = state.samples[Math.floor(Math.random() * state.samples.length)];
  setImage(next, true);
}

function updateCenterFromCanvas(event) {
  const coords = canvasToImageCoords(event.currentTarget, event.clientX, event.clientY);
  if (!coords) return;

  state.cx = coords.x;
  state.cy = coords.y;
  renderAll();
}

function adjustCropSize(delta) {
  const maxSquare = Number(elements.sliders.cropSize.max);
  const next = Math.round(clamp(state.cropSize + delta, 1, maxSquare));
  if (next !== state.cropSize) {
    state.cropSize = next;
    renderAll();
  }
}

function adjustOutputSize(delta) {
  const maxSquare = Number(elements.sliders.outputSize.max);
  const next = Math.round(clamp(state.outputSize + delta, 1, maxSquare));
  if (next !== state.outputSize) {
    state.outputSize = next;
    renderAll();
  }
}

function handleWheel(event) {
  if (!isInteractivePanel(state.activePanel)) return;
  event.preventDefault();

  const delta = event.deltaY < 0 ? 1 : -1;
  if (state.activePanel === 1) {
    adjustCropSize(delta);
  } else {
    adjustOutputSize(delta);
  }
}

function handleArrowKeys(event) {
  if (!isInteractivePanel(state.activePanel)) return;

  if (event.key === "ArrowUp") {
    event.preventDefault();
    adjustCropSize(1);
  } else if (event.key === "ArrowDown") {
    event.preventDefault();
    adjustCropSize(-1);
  } else if (event.key === "ArrowLeft") {
    event.preventDefault();
    adjustOutputSize(-1);
  } else if (event.key === "ArrowRight") {
    event.preventDefault();
    adjustOutputSize(1);
  }
}

function isInteractivePanel(panelIndex) {
  return panelIndex === 1 || panelIndex === 4 || panelIndex === 5;
}

function bindPointerCanvas(canvas, panelIndex) {
  canvas.addEventListener("mouseenter", (event) => {
    state.activePanel = panelIndex;
    updateCenterFromCanvas(event);
  });
  canvas.addEventListener("mouseleave", () => {
    state.activePanel = null;
  });
  canvas.addEventListener("mousemove", updateCenterFromCanvas);
  canvas.addEventListener("wheel", handleWheel, { passive: false });
  canvas.addEventListener("click", (event) => {
    updateCenterFromCanvas(event);
    if (state.activePanel === panelIndex) rememberCurrentCrop();
  });
  canvas.addEventListener("contextmenu", (event) => {
    event.preventDefault();
    updateCenterFromCanvas(event);
    if (state.activePanel === panelIndex) resetMemory();
  });
}

function bindControls() {
  elements.changeImageButton.addEventListener("click", changeImage);
  elements.rememberButton.addEventListener("click", rememberCurrentCrop);
  elements.resetButton.addEventListener("click", resetMemory);

  elements.sliders.centerX.addEventListener("input", (event) => {
    state.cx = Number(event.target.value);
    renderAll();
  });
  elements.sliders.centerY.addEventListener("input", (event) => {
    state.cy = Number(event.target.value);
    renderAll();
  });
  elements.sliders.cropSize.addEventListener("input", (event) => {
    state.cropSize = Number(event.target.value);
    renderAll();
  });
  elements.sliders.outputSize.addEventListener("input", (event) => {
    state.outputSize = Number(event.target.value);
    renderAll();
  });

  elements.blackouts.forEach((checkbox, index) => {
    checkbox.addEventListener("change", () => {
      state.blackout[index] = checkbox.checked;
      renderAll();
    });
  });

  bindPointerCanvas(elements.canvases.original, 1);
  bindPointerCanvas(elements.canvases.reconstructed, 4);
  bindPointerCanvas(elements.canvases.remembered, 5);
  window.addEventListener("keydown", handleArrowKeys);
  window.addEventListener("resize", renderAll);
}

async function loadSamples() {
  const response = await fetch(SAMPLE_PATH);
  if (!response.ok) {
    throw new Error(`Could not load ${SAMPLE_PATH}: ${response.status}`);
  }

  const data = await response.json();
  if (!Array.isArray(data.images) || data.images.length === 0) {
    throw new Error("Sample file does not contain any images.");
  }
  return data.images;
}

async function init() {
  bindControls();
  try {
    state.samples = await loadSamples();
    setImage(state.samples[0], true);
  } catch (error) {
    console.error(error);
    renderAll();
  }
}

init();
