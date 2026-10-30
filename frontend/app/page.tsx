
"use client";

import { useEffect, useRef, useState } from "react";
import {
  Activity,
  UploadCloud,
  Download,
  Github,
  FileText,
  Database,
  Video,
  BrainCircuit,
  AlertTriangle,
  ExternalLink,
  CheckCircle2,
  LoaderCircle,
  Cpu,
  ShieldCheck,
  HardDriveDownload
} from "lucide-react";

import type * as Ort from "onnxruntime-web";

type Detection = {
  class_id: number;
  class_name: string;
  confidence: number;
  bbox: number[];
};

type Prediction = {
  success: boolean;
  filename: string;
  total_detections: number;
  detections: Detection[];
  annotated_image: string;
};

type ModelState = "loading" | "ready" | "error";

const MODEL_URL = "/models/best.onnx";
const MODEL_CACHE_NAME = "roadvision-ai-model-v1";
const MODEL_CACHE_KEY = "/models/best.onnx?v=1";

const INPUT_SIZE = 640;
const CONFIDENCE_THRESHOLD = 0.25;
const IOU_THRESHOLD = 0.45;

const CLASS_NAMES = [
  "crack",
  "pothole",
  "surface_damage",
  "manhole"
];

const CLASS_COLORS = [
  "#22d3ee",
  "#fb923c",
  "#a78bfa",
  "#34d399"
];

const resources = [
  {
    name: "GitHub Repository",
    icon: Github,
    url:
      process.env.NEXT_PUBLIC_GITHUB_URL ||
      "https://github.com/Nvd063/AI-Road-Damage-Detection-System"
  },
  {
    name: "Dataset",
    icon: Database,
    url: process.env.NEXT_PUBLIC_DATASET_URL
  },
  {
    name: "Model Weights",
    icon: BrainCircuit,
    url: process.env.NEXT_PUBLIC_WEIGHTS_URL
  },
  {
    name: "Training Notebook",
    icon: FileText,
    url: process.env.NEXT_PUBLIC_NOTEBOOK_URL
  },
  {
    name: "Project Report",
    icon: FileText,
    url: process.env.NEXT_PUBLIC_REPORT_URL
  },
  {
    name: "Presentation",
    icon: FileText,
    url: process.env.NEXT_PUBLIC_PRESENTATION_URL
  },
  {
    name: "YouTube Demo",
    icon: Video,
    url: process.env.NEXT_PUBLIC_YOUTUBE_URL
  },
  {
    name: "LinkedIn",
    icon: ExternalLink,
    url: process.env.NEXT_PUBLIC_LINKEDIN_URL
  }
];

function intersectionOverUnion(
  a: Detection,
  b: Detection
): number {
  const [ax1, ay1, ax2, ay2] = a.bbox;
  const [bx1, by1, bx2, by2] = b.bbox;

  const x1 = Math.max(ax1, bx1);
  const y1 = Math.max(ay1, by1);
  const x2 = Math.min(ax2, bx2);
  const y2 = Math.min(ay2, by2);

  const intersection =
    Math.max(0, x2 - x1) *
    Math.max(0, y2 - y1);

  const areaA =
    Math.max(0, ax2 - ax1) *
    Math.max(0, ay2 - ay1);

  const areaB =
    Math.max(0, bx2 - bx1) *
    Math.max(0, by2 - by1);

  const union = areaA + areaB - intersection;

  return union > 0 ? intersection / union : 0;
}

function applyNMS(
  detections: Detection[]
): Detection[] {
  const sorted = [...detections].sort(
    (a, b) => b.confidence - a.confidence
  );

  const selected: Detection[] = [];

  for (const detection of sorted) {
    const overlapping = selected.some(
      previous =>
        previous.class_id === detection.class_id &&
        intersectionOverUnion(
          previous,
          detection
        ) > IOU_THRESHOLD
    );

    if (!overlapping) {
      selected.push(detection);
    }

    if (selected.length >= 100) {
      break;
    }
  }

  return selected;
}

async function loadImage(
  file: File
): Promise<HTMLImageElement> {
  const url = URL.createObjectURL(file);

  try {
    const image = new Image();
    image.src = url;

    await image.decode();

    return image;
  } finally {
    URL.revokeObjectURL(url);
  }
}

function preprocessImage(
  image: HTMLImageElement
) {
  const canvas = document.createElement("canvas");

  canvas.width = INPUT_SIZE;
  canvas.height = INPUT_SIZE;

  const ctx = canvas.getContext("2d", {
    willReadFrequently: true
  });

  if (!ctx) {
    throw new Error("Canvas unavailable.");
  }

  const scale = Math.min(
    INPUT_SIZE / image.naturalWidth,
    INPUT_SIZE / image.naturalHeight
  );

  const scaledWidth = Math.round(
    image.naturalWidth * scale
  );

  const scaledHeight = Math.round(
    image.naturalHeight * scale
  );

  const padX = Math.round(
    (INPUT_SIZE - scaledWidth) / 2
  );

  const padY = Math.round(
    (INPUT_SIZE - scaledHeight) / 2
  );

  ctx.fillStyle = "rgb(114,114,114)";

  ctx.fillRect(
    0,
    0,
    INPUT_SIZE,
    INPUT_SIZE
  );

  ctx.drawImage(
    image,
    padX,
    padY,
    scaledWidth,
    scaledHeight
  );

  const pixels = ctx.getImageData(
    0,
    0,
    INPUT_SIZE,
    INPUT_SIZE
  ).data;

  const channelSize = INPUT_SIZE * INPUT_SIZE;

  const input = new Float32Array(
    channelSize * 3
  );

  for (let i = 0; i < channelSize; i++) {
    input[i] = pixels[i * 4] / 255;

    input[channelSize + i] =
      pixels[i * 4 + 1] / 255;

    input[channelSize * 2 + i] =
      pixels[i * 4 + 2] / 255;
  }

  return {
    input,
    scaleX:
      scaledWidth / image.naturalWidth,
    scaleY:
      scaledHeight / image.naturalHeight,
    padX,
    padY
  };
}

function parseOutput(
  output: Ort.Tensor,
  image: HTMLImageElement,
  scaleX: number,
  scaleY: number,
  padX: number,
  padY: number
): Detection[] {
  const dimensions = output.dims.map(Number);

  if (
    dimensions.length !== 3 ||
    dimensions[0] !== 1
  ) {
    throw new Error(
      `Unsupported model output: ${dimensions.join("x")}`
    );
  }

  const channelsFirst =
    dimensions[1] === 8;

  const channelsLast =
    dimensions[2] === 8;

  if (!channelsFirst && !channelsLast) {
    throw new Error(
      `Expected 8 output channels, received ${dimensions.join("x")}`
    );
  }

  const count = channelsFirst
    ? dimensions[2]
    : dimensions[1];

  const data = output.data as Float32Array;

  const getValue = (
    channel: number,
    index: number
  ) =>
    channelsFirst
      ? Number(
          data[channel * count + index]
        )
      : Number(
          data[index * 8 + channel]
        );

  const detections: Detection[] = [];

  for (let i = 0; i < count; i++) {
    let bestClass = 0;
    let bestConfidence = 0;

    for (
      let c = 0;
      c < CLASS_NAMES.length;
      c++
    ) {
      const score = getValue(
        4 + c,
        i
      );

      if (score > bestConfidence) {
        bestConfidence = score;
        bestClass = c;
      }
    }

    if (
      bestConfidence <
      CONFIDENCE_THRESHOLD
    ) {
      continue;
    }

    const cx = getValue(0, i);
    const cy = getValue(1, i);
    const width = getValue(2, i);
    const height = getValue(3, i);

    const x1 = Math.max(
      0,
      (cx - width / 2 - padX) / scaleX
    );

    const y1 = Math.max(
      0,
      (cy - height / 2 - padY) / scaleY
    );

    const x2 = Math.min(
      image.naturalWidth,
      (cx + width / 2 - padX) / scaleX
    );

    const y2 = Math.min(
      image.naturalHeight,
      (cy + height / 2 - padY) / scaleY
    );

    if (
      ![x1, y1, x2, y2].every(
        Number.isFinite
      ) ||
      x2 <= x1 ||
      y2 <= y1
    ) {
      continue;
    }

    detections.push({
      class_id: bestClass,
      class_name:
        CLASS_NAMES[bestClass],
      confidence: bestConfidence,
      bbox: [x1, y1, x2, y2]
    });
  }

  return applyNMS(detections);
}

function drawDetections(
  image: HTMLImageElement,
  detections: Detection[]
): string {
  const canvas = document.createElement("canvas");

  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;

  const ctx = canvas.getContext("2d");

  if (!ctx) {
    throw new Error("Canvas unavailable.");
  }

  ctx.drawImage(image, 0, 0);

  const lineWidth = Math.max(
    2,
    Math.round(
      image.naturalWidth / 400
    )
  );

  const fontSize = Math.max(
    14,
    Math.round(
      image.naturalWidth / 45
    )
  );

  ctx.lineWidth = lineWidth;
  ctx.font = `bold ${fontSize}px Arial`;

  for (const detection of detections) {
    const [x1, y1, x2, y2] =
      detection.bbox;

    const color =
      CLASS_COLORS[
        detection.class_id
      ] || "#22d3ee";

    ctx.strokeStyle = color;

    ctx.strokeRect(
      x1,
      y1,
      x2 - x1,
      y2 - y1
    );

    const label =
      `${detection.class_name.replaceAll("_", " ")} ` +
      `${(detection.confidence * 100).toFixed(1)}%`;

    const textWidth =
      ctx.measureText(label).width;

    const labelHeight =
      fontSize + 12;

    const labelY = Math.max(
      0,
      y1 - labelHeight
    );

    ctx.fillStyle = color;

    ctx.fillRect(
      x1,
      labelY,
      textWidth + 16,
      labelHeight
    );

    ctx.fillStyle = "#07111d";

    ctx.fillText(
      label,
      x1 + 8,
      labelY + fontSize + 2
    );
  }

  return canvas.toDataURL(
    "image/jpeg",
    0.9
  );
}

/*
  Download model once and persist the
  model bytes using the browser Cache API.

  If Cache API is unavailable or storage
  is denied, fall back to normal fetch.
*/
async function getCachedModel(
  onProgress: (message: string) => void
): Promise<ArrayBuffer> {
  let cache: Cache | null = null;

  if ("caches" in window) {
    try {
      cache = await caches.open(
        MODEL_CACHE_NAME
      );

      const cachedResponse =
        await cache.match(
          MODEL_CACHE_KEY
        );

      if (cachedResponse) {
        onProgress(
          "Reading AI model from browser cache..."
        );

        const cachedBytes =
          await cachedResponse.arrayBuffer();

        if (cachedBytes.byteLength > 0) {
          return cachedBytes;
        }
      }
    } catch (error) {
      console.warn(
        "Model cache unavailable:",
        error
      );

      cache = null;
    }
  }

  onProgress(
    "Downloading AI model for the first time..."
  );

  const response = await fetch(
    MODEL_URL,
    {
      cache: "force-cache"
    }
  );

  if (!response.ok) {
    throw new Error(
      `Model download failed: HTTP ${response.status}`
    );
  }

  const modelBytes =
    await response.arrayBuffer();

  if (modelBytes.byteLength === 0) {
    throw new Error(
      "Downloaded model is empty."
    );
  }

  if (cache) {
    try {
      const cacheResponse = new Response(
        modelBytes.slice(0),
        {
          headers: {
            "Content-Type":
              "application/octet-stream"
          }
        }
      );

      await cache.put(
        MODEL_CACHE_KEY,
        cacheResponse
      );
    } catch (error) {
      console.warn(
        "Could not persist model cache:",
        error
      );
    }
  }

  return modelBytes;
}

export default function Home() {
  const [file, setFile] =
    useState<File | null>(null);

  const [preview, setPreview] =
    useState("");

  const [result, setResult] =
    useState<Prediction | null>(null);

  const [loading, setLoading] =
    useState(false);

  const [error, setError] =
    useState("");

  const [modelState, setModelState] =
    useState<ModelState>("loading");

  const [modelProgress, setModelProgress] =
    useState(
      "Preparing AI model..."
    );

  const [predictionProgress, setPredictionProgress] =
    useState("");

  const [inferenceTime, setInferenceTime] =
    useState<number | null>(null);

  const [modelLoadTime, setModelLoadTime] =
    useState<number | null>(null);

  const sessionRef =
    useRef<Ort.InferenceSession | null>(
      null
    );

  const ortRef =
    useRef<typeof Ort | null>(
      null
    );

  const modelPromiseRef =
    useRef<
      Promise<{
        session: Ort.InferenceSession;
        ort: typeof Ort;
      }> | null
    >(null);

  const busyRef = useRef(false);

  /*
    Single-flight initialization:
    preload and Run AI Detection share
    exactly the same loading promise.
  */
  function initializeModel() {
    if (modelPromiseRef.current) {
      return modelPromiseRef.current;
    }

    const promise = (async () => {
      const start = performance.now();

      setModelState("loading");

      setModelProgress(
        "Initializing ONNX Runtime..."
      );

      const ort =
        await import("onnxruntime-web");

      ort.env.wasm.numThreads = 1;
      ort.env.wasm.simd = true;

      ort.env.wasm.wasmPaths =
        "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/";

      const modelBytes =
        await getCachedModel(
          setModelProgress
        );

      setModelProgress(
        "Initializing AI inference session..."
      );

      const session =
        await ort.InferenceSession.create(
          modelBytes,
          {
            executionProviders: [
              "wasm"
            ],
            graphOptimizationLevel:
              "all"
          }
        );

      sessionRef.current = session;
      ortRef.current = ort;

      const elapsed =
        performance.now() - start;

      setModelLoadTime(elapsed);
      setModelState("ready");
      setModelProgress(
        "AI model ready for prediction."
      );

      return {
        session,
        ort
      };
    })();

    modelPromiseRef.current = promise;

    promise.catch(error => {
      console.error(
        "Model preload failed:",
        error
      );

      setModelState("error");

      setModelProgress(
        error instanceof Error
          ? error.message
          : "Model initialization failed."
      );

      modelPromiseRef.current = null;
    });

    return promise;
  }

  /*
    Automatically preload the model
    as soon as the page is mounted.
  */
  useEffect(() => {
    void initializeModel().catch(
      () => undefined
    );
  }, []);

  useEffect(() => {
    if (!file) {
      setPreview("");
      return;
    }

    const url =
      URL.createObjectURL(file);

    setPreview(url);

    return () =>
      URL.revokeObjectURL(url);
  }, [file]);

  function chooseFile(
    selected?: File
  ) {
    if (busyRef.current) {
      return;
    }

    setResult(null);
    setError("");
    setInferenceTime(null);

    if (!selected) return;

    if (
      ![
        "image/jpeg",
        "image/png",
        "image/webp"
      ].includes(selected.type)
    ) {
      setError(
        "Only JPG, PNG or WEBP images are allowed."
      );
      return;
    }

    if (
      selected.size >
      10 * 1024 * 1024
    ) {
      setError(
        "Maximum image size is 10 MB."
      );
      return;
    }

    setFile(selected);
  }

  async function analyze() {
    if (
      !file ||
      busyRef.current
    ) {
      return;
    }

    busyRef.current = true;

    setLoading(true);
    setError("");
    setResult(null);
    setInferenceTime(null);

    try {
      setPredictionProgress(
        "Waiting for AI model..."
      );

      /*
        If preloading has finished,
        this resolves immediately.

        If still loading, prediction
        waits for the same session.
      */
      const {
        session,
        ort
      } = await initializeModel();

      setPredictionProgress(
        "Preparing road image..."
      );

      const image =
        await loadImage(file);

      if (
        image.naturalWidth *
          image.naturalHeight >
        20_000_000
      ) {
        throw new Error(
          "Maximum image resolution is 20 megapixels."
        );
      }

      const processed =
        preprocessImage(image);

      const inputTensor =
        new ort.Tensor(
          "float32",
          processed.input,
          [
            1,
            3,
            INPUT_SIZE,
            INPUT_SIZE
          ]
        );

      setPredictionProgress(
        "Running YOLO11s inference..."
      );

      const start =
        performance.now();

      const outputs =
        await session.run({
          [session.inputNames[0]]:
            inputTensor
        });

      const elapsed =
        performance.now() - start;

      const output =
        outputs[
          session.outputNames[0]
        ];

      if (!output) {
        throw new Error(
          "AI model returned no output."
        );
      }

      setPredictionProgress(
        "Processing bounding boxes..."
      );

      const detections =
        parseOutput(
          output,
          image,
          processed.scaleX,
          processed.scaleY,
          processed.padX,
          processed.padY
        );

      const annotatedImage =
        drawDetections(
          image,
          detections
        );

      setResult({
        success: true,
        filename: file.name,
        total_detections:
          detections.length,
        detections,
        annotated_image:
          annotatedImage
      });

      setInferenceTime(elapsed);
      setPredictionProgress("");
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "AI inference failed."
      );

      setPredictionProgress("");
    } finally {
      busyRef.current = false;
      setLoading(false);
    }
  }

  function downloadReport() {
    if (!result) return;

    const report = [
      "ROADVISION AI - DETECTION REPORT",
      "================================",
      `Image: ${result.filename}`,
      `Date: ${new Date().toISOString()}`,
      "Model: YOLO11s (ONNX)",
      `Total detections: ${result.total_detections}`,
      "",
      ...result.detections.map(
        (d, i) =>
          `${i + 1}. ${d.class_name} | ` +
          `Confidence: ${(d.confidence * 100).toFixed(1)}% | ` +
          `Box: ${d.bbox.map(
            v => v.toFixed(1)
          ).join(", ")}`
      ),
      "",
      "AI-generated results require human verification."
    ].join("\n");

    const blob = new Blob(
      [report],
      {
        type: "text/plain"
      }
    );

    const url =
      URL.createObjectURL(blob);

    const anchor =
      document.createElement("a");

    anchor.href = url;
    anchor.download =
      "road-damage-report.txt";

    anchor.click();

    URL.revokeObjectURL(url);
  }

  function downloadAnnotatedImage() {
    if (!result) return;

    const anchor =
      document.createElement("a");

    anchor.href =
      result.annotated_image;

    anchor.download =
      "road-damage-detection.jpg";

    anchor.click();
  }

  return (
    <main className="max-w-7xl mx-auto px-5 py-8 space-y-8">
      <header className="flex flex-wrap justify-between items-center gap-4 border-b border-slate-800 pb-6">
        <div className="flex items-center gap-3">
          <BrainCircuit
            className="text-cyan-400"
            size={34}
          />

          <div>
            <h1 className="text-2xl font-bold">
              RoadVision AI
            </h1>

            <p className="text-slate-400 text-sm">
              Automated Road Damage Detection & Reporting
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 bg-slate-800 rounded-full px-4 py-2 text-sm">
          {modelState === "ready" ? (
            <CheckCircle2
              size={16}
              className="text-green-400"
            />
          ) : modelState === "loading" ? (
            <LoaderCircle
              size={16}
              className="text-cyan-400 animate-spin"
            />
          ) : (
            <AlertTriangle
              size={16}
              className="text-red-400"
            />
          )}

          <span>AI Model:</span>

          <span
            className={
              modelState === "ready"
                ? "text-green-400 font-semibold"
                : modelState === "error"
                  ? "text-red-400 font-semibold"
                  : "text-cyan-400 font-semibold"
            }
          >
            {modelState === "ready"
              ? "Ready"
              : modelState === "loading"
                ? "Loading..."
                : "Unavailable"}
          </span>
        </div>
      </header>

      <section className="py-7">
        <p className="text-cyan-400 font-semibold mb-3">
          DEEP LEARNING PROJECT
        </p>

        <h2 className="text-4xl md:text-5xl font-bold max-w-3xl leading-tight">
          Smarter Road Inspection Powered by Artificial Intelligence
        </h2>

        <p className="text-slate-400 mt-5 max-w-2xl">
          Detect cracks, potholes, surface damage and
          manholes using a custom-trained YOLO11s
          model running directly in your browser.
        </p>

        <div className="flex items-center gap-2 mt-5 text-green-400 text-sm">
          <ShieldCheck size={18} />

          Browser-based AI · No image upload to a server
        </div>
      </section>

      <section className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          ["Model", "YOLO11s"],
          ["Damage Classes", "4"],
          ["Test mAP50", "44.97%"],
          ["Test Recall", "49.35%"]
        ].map(([label, value]) => (
          <div
            className="panel"
            key={label}
          >
            <p className="text-slate-400 text-sm">
              {label}
            </p>

            <p className="text-2xl font-bold mt-2">
              {value}
            </p>
          </div>
        ))}
      </section>

      <section className="panel space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <HardDriveDownload
              size={22}
              className={
                modelState === "ready"
                  ? "text-green-400"
                  : "text-cyan-400"
              }
            />

            <div>
              <h3 className="font-semibold">
                AI Model Initialization
              </h3>

              <p className="text-slate-400 text-sm">
                {modelProgress}
              </p>
            </div>
          </div>

          {modelState === "ready" &&
            modelLoadTime !== null && (
              <span className="text-xs text-green-400">
                Ready in{" "}
                {(modelLoadTime / 1000).toFixed(1)}s
              </span>
            )}
        </div>

        {modelState === "loading" && (
          <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
            <div className="h-full bg-cyan-400 w-2/3 animate-pulse rounded-full" />
          </div>
        )}

        {modelState === "error" && (
          <button
            className="btn"
            onClick={() => {
              void initializeModel().catch(
                () => undefined
              );
            }}
          >
            Retry Model Loading
          </button>
        )}

        <p className="text-slate-500 text-xs">
          The AI model loads automatically in the
          background. Browser caching helps speed
          up subsequent visits.
        </p>
      </section>

      <section className="grid lg:grid-cols-2 gap-6">
        <div className="panel space-y-5">
          <h3 className="text-xl font-bold flex gap-2 items-center">
            <UploadCloud className="text-cyan-400" />
            AI Damage Detection
          </h3>

          <label
            className="border-2 border-dashed border-slate-600 rounded-xl p-8 text-center block cursor-pointer hover:border-cyan-400 transition-colors"
            onDragOver={e =>
              e.preventDefault()
            }
            onDrop={e => {
              e.preventDefault();

              chooseFile(
                e.dataTransfer.files[0]
              );
            }}
          >
            <UploadCloud
              className="mx-auto text-cyan-400 mb-3"
              size={36}
            />

            <p className="break-all">
              {file
                ? file.name
                : "Drop an image here or click to browse"}
            </p>

            <p className="text-slate-500 text-sm mt-2">
              JPG, PNG, WEBP · Max 10 MB
            </p>

            <input
              type="file"
              accept="image/jpeg,image/png,image/webp"
              className="hidden"
              onChange={e =>
                chooseFile(
                  e.target.files?.[0]
                )
              }
            />
          </label>

          {preview && (
            <img
              src={preview}
              alt="Uploaded road"
              className="rounded-xl w-full"
            />
          )}

          <button
            className="btn w-full flex items-center justify-center gap-2"
            disabled={
              !file ||
              loading ||
              modelState === "error"
            }
            onClick={analyze}
          >
            {loading && (
              <LoaderCircle
                size={19}
                className="animate-spin"
              />
            )}

            {loading
              ? "Analyzing Image..."
              : modelState === "loading"
                ? "Run AI Detection (Model Loading)"
                : "Run AI Detection"}
          </button>

          {loading && (
            <div className="bg-slate-800 rounded-xl p-4 space-y-3">
              <p className="text-cyan-400 text-sm">
                {predictionProgress}
              </p>

              <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                <div className="h-full bg-cyan-400 rounded-full animate-pulse w-2/3" />
              </div>
            </div>
          )}

          {error && (
            <div className="bg-red-950/40 border border-red-800 rounded-xl p-4 text-red-300 text-sm">
              {error}
            </div>
          )}
        </div>

        <div className="panel space-y-5">
          <h3 className="text-xl font-bold flex gap-2 items-center">
            <Activity className="text-cyan-400" />
            Detection Results
          </h3>

          {!result ? (
            <div className="min-h-64 flex items-center justify-center text-slate-500 text-center">
              Upload an image and run detection
              to view AI predictions.
            </div>
          ) : (
            <>
              <img
                src={result.annotated_image}
                alt="AI annotated detection"
                className="w-full rounded-xl"
              />

              <div className="flex flex-wrap justify-between items-center gap-3">
                <p className="font-semibold">
                  {result.total_detections} detections found
                </p>

                {inferenceTime !== null && (
                  <span className="text-xs text-slate-400">
                    Inference:{" "}
                    {inferenceTime.toFixed(0)} ms
                  </span>
                )}
              </div>

              {result.total_detections === 0 && (
                <p className="text-slate-400 text-sm">
                  No road damage detected above
                  the confidence threshold.
                </p>
              )}

              <div className="space-y-2 max-h-64 overflow-y-auto">
                {result.detections.map(
                  (d, i) => (
                    <div
                      key={i}
                      className="flex justify-between gap-3 bg-slate-800 rounded-lg p-3 text-sm"
                    >
                      <span className="capitalize">
                        {d.class_name.replaceAll(
                          "_",
                          " "
                        )}
                      </span>

                      <span className="text-cyan-400 font-semibold">
                        {(d.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  )
                )}
              </div>

              <div className="grid sm:grid-cols-2 gap-3">
                <button
                  className="btn flex items-center justify-center gap-2"
                  onClick={downloadReport}
                >
                  <Download size={18} />
                  Report
                </button>

                <button
                  className="btn flex items-center justify-center gap-2"
                  onClick={downloadAnnotatedImage}
                >
                  <Download size={18} />
                  Image
                </button>
              </div>
            </>
          )}
        </div>
      </section>

      <section className="panel">
        <h3 className="text-xl font-bold mb-5">
          Supported Detection Classes
        </h3>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {CLASS_NAMES.map(
            (name, i) => (
              <div
                key={name}
                className="bg-slate-800 rounded-xl p-5 text-center"
              >
                <AlertTriangle
                  className="mx-auto mb-3"
                  style={{
                    color:
                      CLASS_COLORS[i]
                  }}
                />

                <p className="font-semibold capitalize">
                  {name.replaceAll(
                    "_",
                    " "
                  )}
                </p>
              </div>
            )
          )}
        </div>
      </section>

      <section className="panel">
        <h3 className="text-xl font-bold mb-5">
          Project Resources
        </h3>

        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {resources.map(resource => {
            const Icon =
              resource.icon;

            return resource.url ? (
              <a
                key={resource.name}
                className="resource flex justify-center items-center gap-2"
                href={resource.url}
                target="_blank"
                rel="noopener noreferrer"
              >
                <Icon size={18} />

                {resource.name}

                <ExternalLink size={14} />
              </a>
            ) : (
              <div
                key={resource.name}
                className="resource opacity-50 flex justify-center items-center gap-2"
              >
                <Icon size={18} />

                {resource.name} — Coming Soon
              </div>
            );
          })}
        </div>
      </section>

      <footer className="text-center text-slate-500 text-sm py-6">
        RoadVision AI · Academic Deep Learning Project ·
        Predictions require human verification
      </footer>
    </main>
  );
}
