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
  HardDriveDownload,
  ArrowRight,
  GraduationCap,
  Layers3,
  ScanSearch,
  Sparkles,
  LockKeyhole,
  ChartNoAxesCombined,
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

const CLASS_NAMES = ["crack", "pothole", "surface_damage", "manhole"];

const CLASS_COLORS = ["#22d3ee", "#fb923c", "#a78bfa", "#34d399"];

const resources = [
  {
    name: "Source code",
    detail: "GitHub · ZIP archive",
    logo: "github",
    url: "https://github.com/Nvd063/AI-Road-Damage-Detection-System/archive/refs/heads/main.zip",
    type: "download",
  },
  {
    name: "Road dataset",
    detail: "Google Drive · dataset",
    logo: "drive",
    url: "https://drive.google.com/uc?export=download&id=17OGIQSzPAwrZMbajeouU4X-H51O-x9hC",
    type: "download",
  },
  {
    name: "Trained weights",
    detail: "Google Drive · model file",
    logo: "drive",
    url: "https://drive.google.com/uc?export=download&id=1vUhbX0Mf_v2rkpsSjPPjLj-6iq7sNnHT",
    type: "download",
  },
  {
    name: "Training notebook",
    detail: "Google Colab · notebook",
    logo: "colab",
    url: "https://drive.google.com/uc?export=download&id=1ZtWRj32noe_oMdmB0nrSaw2r_zcnJpxX",
    type: "download",
  },
  {
    name: "Project report",
    detail: "Google Drive · document",
    logo: "drive",
    url: "https://drive.google.com/uc?export=download&id=1fQqSJgW_lyzJjVNoAhkJ_Lk0t3Q3V8tO",
    type: "download",
  },
  {
    name: "Presentation",
    detail: "Google Slides · PPTX",
    logo: "slides",
    url: "https://docs.google.com/presentation/d/16q5UMqWouLrgCrICeUJGqwXeEf9ua02o/export/pptx",
    type: "download",
  },
] as const;
const YOUTUBE_URL = "https://youtu.be/thgqco_hMpY";
const classes = [
  { name: "Cracks", color: "#22d3ee", index: "01" },
  { name: "Potholes", color: "#fb923c", index: "02" },
  { name: "Surface damage", color: "#a78bfa", index: "03" },
  { name: "Manholes", color: "#34d399", index: "04" },
];
function BrandLogo({ brand }: { brand: string }) {
  const urls: Record<string, string> = {
    github: "https://cdn.simpleicons.org/github/ffffff",
    drive: "https://cdn.simpleicons.org/googledrive",
    colab: "https://cdn.simpleicons.org/googlecolab",
    slides: "https://cdn.simpleicons.org/googleslides",
  };
  return (
    <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-white/10">
      <img
        src={urls[brand]}
        alt={`${brand} logo`}
        width={27}
        height={27}
        loading="lazy"
        onError={(e) => {
          e.currentTarget.style.display = "none";
        }}
      />
    </span>
  );
}

function intersectionOverUnion(
  a: Detection,

  b: Detection,
): number {
  const [ax1, ay1, ax2, ay2] = a.bbox;

  const [bx1, by1, bx2, by2] = b.bbox;

  const x1 = Math.max(ax1, bx1);

  const y1 = Math.max(ay1, by1);

  const x2 = Math.min(ax2, bx2);

  const y2 = Math.min(ay2, by2);

  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);

  const areaA = Math.max(0, ax2 - ax1) * Math.max(0, ay2 - ay1);

  const areaB = Math.max(0, bx2 - bx1) * Math.max(0, by2 - by1);

  const union = areaA + areaB - intersection;

  return union > 0 ? intersection / union : 0;
}

function applyNMS(detections: Detection[]): Detection[] {
  const sorted = [...detections].sort((a, b) => b.confidence - a.confidence);

  const selected: Detection[] = [];

  for (const detection of sorted) {
    const overlapping = selected.some(
      (previous) =>
        previous.class_id === detection.class_id &&
        intersectionOverUnion(
          previous,

          detection,
        ) > IOU_THRESHOLD,
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

async function loadImage(file: File): Promise<HTMLImageElement> {
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

function preprocessImage(image: HTMLImageElement) {
  const canvas = document.createElement("canvas");

  canvas.width = INPUT_SIZE;

  canvas.height = INPUT_SIZE;

  const ctx = canvas.getContext("2d", {
    willReadFrequently: true,
  });

  if (!ctx) {
    throw new Error("Canvas unavailable.");
  }

  const scale = Math.min(
    INPUT_SIZE / image.naturalWidth,

    INPUT_SIZE / image.naturalHeight,
  );

  const scaledWidth = Math.round(image.naturalWidth * scale);

  const scaledHeight = Math.round(image.naturalHeight * scale);

  const padX = Math.round((INPUT_SIZE - scaledWidth) / 2);

  const padY = Math.round((INPUT_SIZE - scaledHeight) / 2);

  ctx.fillStyle = "rgb(114,114,114)";

  ctx.fillRect(
    0,

    0,

    INPUT_SIZE,

    INPUT_SIZE,
  );

  ctx.drawImage(
    image,

    padX,

    padY,

    scaledWidth,

    scaledHeight,
  );

  const pixels = ctx.getImageData(
    0,

    0,

    INPUT_SIZE,

    INPUT_SIZE,
  ).data;

  const channelSize = INPUT_SIZE * INPUT_SIZE;

  const input = new Float32Array(channelSize * 3);

  for (let i = 0; i < channelSize; i++) {
    input[i] = pixels[i * 4] / 255;

    input[channelSize + i] = pixels[i * 4 + 1] / 255;

    input[channelSize * 2 + i] = pixels[i * 4 + 2] / 255;
  }

  return {
    input,

    scaleX: scaledWidth / image.naturalWidth,

    scaleY: scaledHeight / image.naturalHeight,

    padX,

    padY,
  };
}

function parseOutput(
  output: Ort.Tensor,

  image: HTMLImageElement,

  scaleX: number,

  scaleY: number,

  padX: number,

  padY: number,
): Detection[] {
  const dimensions = output.dims.map(Number);

  if (dimensions.length !== 3 || dimensions[0] !== 1) {
    throw new Error(`Unsupported model output: ${dimensions.join("x")}`);
  }

  const channelsFirst = dimensions[1] === 8;

  const channelsLast = dimensions[2] === 8;

  if (!channelsFirst && !channelsLast) {
    throw new Error(
      `Expected 8 output channels, received ${dimensions.join("x")}`,
    );
  }

  const count = channelsFirst ? dimensions[2] : dimensions[1];

  const data = output.data as Float32Array;

  const getValue = (
    channel: number,

    index: number,
  ) =>
    channelsFirst
      ? Number(data[channel * count + index])
      : Number(data[index * 8 + channel]);

  const detections: Detection[] = [];

  for (let i = 0; i < count; i++) {
    let bestClass = 0;

    let bestConfidence = 0;

    for (let c = 0; c < CLASS_NAMES.length; c++) {
      const score = getValue(
        4 + c,

        i,
      );

      if (score > bestConfidence) {
        bestConfidence = score;

        bestClass = c;
      }
    }

    if (bestConfidence < CONFIDENCE_THRESHOLD) {
      continue;
    }

    const cx = getValue(0, i);

    const cy = getValue(1, i);

    const width = getValue(2, i);

    const height = getValue(3, i);

    const x1 = Math.max(
      0,

      (cx - width / 2 - padX) / scaleX,
    );

    const y1 = Math.max(
      0,

      (cy - height / 2 - padY) / scaleY,
    );

    const x2 = Math.min(
      image.naturalWidth,

      (cx + width / 2 - padX) / scaleX,
    );

    const y2 = Math.min(
      image.naturalHeight,

      (cy + height / 2 - padY) / scaleY,
    );

    if (![x1, y1, x2, y2].every(Number.isFinite) || x2 <= x1 || y2 <= y1) {
      continue;
    }

    detections.push({
      class_id: bestClass,

      class_name: CLASS_NAMES[bestClass],

      confidence: bestConfidence,

      bbox: [x1, y1, x2, y2],
    });
  }

  return applyNMS(detections);
}

function drawDetections(
  image: HTMLImageElement,

  detections: Detection[],
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

    Math.round(image.naturalWidth / 400),
  );

  const fontSize = Math.max(
    14,

    Math.round(image.naturalWidth / 45),
  );

  ctx.lineWidth = lineWidth;

  ctx.font = `bold ${fontSize}px Arial`;

  for (const detection of detections) {
    const [x1, y1, x2, y2] = detection.bbox;

    const color = CLASS_COLORS[detection.class_id] || "#22d3ee";

    ctx.strokeStyle = color;

    ctx.strokeRect(
      x1,

      y1,

      x2 - x1,

      y2 - y1,
    );

    const label =
      `${detection.class_name.replaceAll("_", " ")} ` +
      `${(detection.confidence * 100).toFixed(1)}%`;

    const textWidth = ctx.measureText(label).width;

    const labelHeight = fontSize + 12;

    const labelY = Math.max(
      0,

      y1 - labelHeight,
    );

    ctx.fillStyle = color;

    ctx.fillRect(
      x1,

      labelY,

      textWidth + 16,

      labelHeight,
    );

    ctx.fillStyle = "#07111d";

    ctx.fillText(
      label,

      x1 + 8,

      labelY + fontSize + 2,
    );
  }

  return canvas.toDataURL(
    "image/jpeg",

    0.9,
  );
}

async function getCachedModel(
  onProgress: (message: string) => void,
): Promise<ArrayBuffer> {
  let cache: Cache | null = null;

  if ("caches" in window) {
    try {
      cache = await caches.open(MODEL_CACHE_NAME);

      const cachedResponse = await cache.match(MODEL_CACHE_KEY);

      if (cachedResponse) {
        onProgress("Reading AI model from browser cache...");

        const cachedBytes = await cachedResponse.arrayBuffer();

        if (cachedBytes.byteLength > 0) {
          return cachedBytes;
        }
      }
    } catch (error) {
      console.warn(
        "Model cache unavailable:",

        error,
      );

      cache = null;
    }
  }

  onProgress("Downloading AI model for the first time...");

  const response = await fetch(
    MODEL_URL,

    {
      cache: "force-cache",
    },
  );

  if (!response.ok) {
    throw new Error(`Model download failed: HTTP ${response.status}`);
  }

  const modelBytes = await response.arrayBuffer();

  if (modelBytes.byteLength === 0) {
    throw new Error("Downloaded model is empty.");
  }

  if (cache) {
    try {
      const cacheResponse = new Response(
        modelBytes.slice(0),

        {
          headers: {
            "Content-Type": "application/octet-stream",
          },
        },
      );

      await cache.put(
        MODEL_CACHE_KEY,

        cacheResponse,
      );
    } catch (error) {
      console.warn(
        "Could not persist model cache:",

        error,
      );
    }
  }

  return modelBytes;
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);

  const [preview, setPreview] = useState("");

  const [result, setResult] = useState<Prediction | null>(null);

  const [loading, setLoading] = useState(false);

  const [error, setError] = useState("");

  const [modelState, setModelState] = useState<ModelState>("loading");

  const [modelProgress, setModelProgress] = useState("Preparing AI model...");

  const [predictionProgress, setPredictionProgress] = useState("");

  const [inferenceTime, setInferenceTime] = useState<number | null>(null);

  const [modelLoadTime, setModelLoadTime] = useState<number | null>(null);

  const sessionRef = useRef<Ort.InferenceSession | null>(null);

  const ortRef = useRef<typeof Ort | null>(null);

  const modelPromiseRef = useRef<Promise<{
    session: Ort.InferenceSession;

    ort: typeof Ort;
  }> | null>(null);

  const busyRef = useRef(false);

  function initializeModel() {
    if (modelPromiseRef.current) {
      return modelPromiseRef.current;
    }

    const promise = (async () => {
      const start = performance.now();

      setModelState("loading");

      setModelProgress("Initializing ONNX Runtime...");

      const ort = await import("onnxruntime-web");

      ort.env.wasm.numThreads = 1;

      ort.env.wasm.simd = true;

      ort.env.wasm.wasmPaths =
        "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/";

      const modelBytes = await getCachedModel(setModelProgress);

      setModelProgress("Initializing AI inference session...");

      const session = await ort.InferenceSession.create(
        modelBytes,

        {
          executionProviders: ["wasm"],

          graphOptimizationLevel: "all",
        },
      );

      sessionRef.current = session;

      ortRef.current = ort;

      const elapsed = performance.now() - start;

      setModelLoadTime(elapsed);

      setModelState("ready");

      setModelProgress("AI model ready for prediction.");

      return {
        session,

        ort,
      };
    })();

    modelPromiseRef.current = promise;

    promise.catch((error) => {
      console.error(
        "Model preload failed:",

        error,
      );

      setModelState("error");

      setModelProgress(
        error instanceof Error ? error.message : "Model initialization failed.",
      );

      modelPromiseRef.current = null;
    });

    return promise;
  }

  useEffect(() => {
    void initializeModel().catch(() => undefined);
  }, []);

  useEffect(() => {
    if (!file) {
      setPreview("");

      return;
    }

    const url = URL.createObjectURL(file);

    setPreview(url);

    return () => URL.revokeObjectURL(url);
  }, [file]);

  function chooseFile(selected?: File) {
    if (busyRef.current) {
      return;
    }

    setResult(null);

    setError("");

    setInferenceTime(null);

    if (!selected) return;

    if (!["image/jpeg", "image/png", "image/webp"].includes(selected.type)) {
      setError("Only JPG, PNG or WEBP images are allowed.");

      return;
    }

    if (selected.size > 10 * 1024 * 1024) {
      setError("Maximum image size is 10 MB.");

      return;
    }

    setFile(selected);
  }

  async function analyze() {
    if (!file || busyRef.current) {
      return;
    }

    busyRef.current = true;

    setLoading(true);

    setError("");

    setResult(null);

    setInferenceTime(null);

    try {
      setPredictionProgress("Waiting for AI model...");

      const {
        session,

        ort,
      } = await initializeModel();

      setPredictionProgress("Preparing road image...");

      const image = await loadImage(file);

      if (image.naturalWidth * image.naturalHeight > 20_000_000) {
        throw new Error("Maximum image resolution is 20 megapixels.");
      }

      const processed = preprocessImage(image);

      const inputTensor = new ort.Tensor(
        "float32",

        processed.input,

        [1, 3, INPUT_SIZE, INPUT_SIZE],
      );

      setPredictionProgress("Running YOLO11s inference...");

      const start = performance.now();

      const outputs = await session.run({
        [session.inputNames[0]]: inputTensor,
      });

      const elapsed = performance.now() - start;

      const output = outputs[session.outputNames[0]];

      if (!output) {
        throw new Error("AI model returned no output.");
      }

      setPredictionProgress("Processing bounding boxes...");

      const detections = parseOutput(
        output,

        image,

        processed.scaleX,

        processed.scaleY,

        processed.padX,

        processed.padY,
      );

      const annotatedImage = drawDetections(
        image,

        detections,
      );

      setResult({
        success: true,

        filename: file.name,

        total_detections: detections.length,

        detections,

        annotated_image: annotatedImage,
      });

      setInferenceTime(elapsed);

      setPredictionProgress("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "AI inference failed.");

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
          `Box: ${d.bbox.map((v) => v.toFixed(1)).join(", ")}`,
      ),

      "",

      "AI-generated results require human verification.",
    ].join("\n");

    const blob = new Blob(
      [report],

      {
        type: "text/plain",
      },
    );

    const url = URL.createObjectURL(blob);

    const anchor = document.createElement("a");

    anchor.href = url;

    anchor.download = "road-damage-report.txt";

    anchor.click();

    URL.revokeObjectURL(url);
  }

  function downloadAnnotatedImage() {
    if (!result) return;

    const anchor = document.createElement("a");

    anchor.href = result.annotated_image;

    anchor.download = "road-damage-detection.jpg";

    anchor.click();
  }

  const [activeTab, setActiveTab] = useState<"detect" | "results">("detect");
  const [showDetails, setShowDetails] = useState(false);
  const [downloadNotice, setDownloadNotice] = useState("");
  function startResourceDownload(url: string, name: string) {
    setDownloadNotice(
      `Opening ${name} download. The file host may ask for confirmation or permission.`,
    );
    window.open(url, "_blank", "noopener,noreferrer");
  }
  return (
    <main
      id="top"
      className="min-h-screen bg-[#050913] text-white selection:bg-cyan-400/30"
    >
      <div
        className="pointer-events-none fixed inset-0 opacity-50"
        style={{
          backgroundImage:
            "radial-gradient(ellipse at 10% 0%,rgba(18,103,154,.23),transparent 42%),radial-gradient(ellipse at 90% 40%,rgba(65,42,160,.16),transparent 40%)",
        }}
      />
      <div className="relative mx-auto max-w-[1440px] px-4 pb-8 pt-4 sm:px-7 lg:px-10">
        <header className="sticky top-3 z-30 mb-5 flex items-center justify-between gap-3 rounded-2xl border border-white/10 bg-[#091121]/90 px-4 py-3 shadow-2xl backdrop-blur-xl sm:px-6">
          <a href="#top" className="flex min-w-0 items-center gap-2.5">
            <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-cyan-400 text-[#06111a]">
              <ScanSearch size={23} />
            </span>
            <span className="truncate text-base font-black tracking-tight sm:text-xl">
              ROADVISION <span className="text-cyan-400">AI</span>
            </span>
          </a>
          <nav className="hidden items-center gap-6 text-xs font-bold text-slate-300 lg:flex">
            <a href="#experience" className="hover:text-cyan-300">
              Experience
            </a>
            <a href="#showcase" className="hover:text-cyan-300">
              Presentation
            </a>
            <a href="#resources" className="hover:text-cyan-300">
              Resources
            </a>
            <a href="#academic" className="hover:text-cyan-300">
              Project
            </a>
          </nav>
          <a
            href="#experience"
            className="inline-flex shrink-0 items-center gap-2 rounded-xl bg-cyan-400 px-3 py-2.5 text-xs font-black text-[#06111a] sm:px-5"
          >
            <ScanSearch size={15} /> Launch AI <ArrowRight size={14} />
          </a>
        </header>

        <section
          id="academic"
          className="mt-5 scroll-mt-24 rounded-[22px] border border-white/10 bg-[#0b1424] px-4 py-4 sm:px-7"
        >
          <button
            type="button"
            onClick={() => setShowDetails((v) => !v)}
            aria-expanded={showDetails}
            aria-controls="academic-details-panel"
            className="group flex w-full items-center justify-between gap-3 text-left"
          >
            <span className="flex items-center gap-3">
              <GraduationCap className="text-cyan-300" size={23} />
              <span>
                <b className="block text-sm sm:text-base">
                  Academic project details
                </b>
                <span className="text-[11px] text-slate-500">
                  Superior University · Deep Learning
                </span>
              </span>
            </span>
            <span className="flex shrink-0 items-center gap-2 text-xs font-bold text-cyan-300">
              <span>{showDetails ? "Collapse" : "Expand"}</span>
              <span
                aria-hidden="true"
                className={`flex h-7 w-7 items-center justify-center rounded-full border border-cyan-400/25 bg-cyan-400/10 text-lg transition-transform duration-500 ease-in-out motion-reduce:transition-none ${showDetails ? "rotate-180" : "rotate-0"}`}
              >
                {showDetails ? "−" : "+"}
              </span>
            </span>
          </button>
          <div
            id="academic-details-panel"
            aria-hidden={!showDetails}
            inert={!showDetails}
            className={`grid transition-[grid-template-rows,opacity] duration-500 ease-in-out motion-reduce:transition-none ${showDetails ? "grid-rows-[1fr] opacity-100" : "grid-rows-[0fr] opacity-0"}`}
          >
            <div className="min-h-0 overflow-hidden">
              <div className="mt-5 grid gap-3 border-t border-white/10 pt-5 sm:grid-cols-2 lg:grid-cols-3">
                {[
                  ["Student", "Naveed Ahmed"],
                  ["Roll no.", "SU92-BSCSM-F23-600"],
                  ["Class", "BSCS (7A)"],
                  ["Course", "Deep Learning"],
                  ["Instructor", "Dr. Munir Ahmad"],
                  ["Institution", "Superior University"],
                ].map(([label, value], index) => (
                  <div
                    key={label}
                    className={`rounded-xl bg-white/5 p-3 transition-[transform,opacity] duration-500 ease-out motion-reduce:transition-none ${showDetails ? "translate-y-0 opacity-100" : "translate-y-2 opacity-0"}`}
                    style={{ transitionDelay: showDetails ? `${index * 45}ms` : "0ms" }}
                  >
                    <p className="text-[10px] uppercase tracking-widest text-slate-500">
                      {label}
                    </p>
                    <p className="mt-1 text-sm font-bold">{value}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        <section className="relative overflow-hidden rounded-[26px] border border-white/10 bg-[#0b1424] p-5 sm:p-8 lg:p-11">
          <div className="pointer-events-none absolute -right-20 -top-32 h-96 w-96 rounded-full bg-cyan-500/10 blur-3xl" />
          <div className="relative grid items-center gap-8 lg:grid-cols-[1.05fr_.95fr]">
            <div>
              <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-cyan-400/25 bg-cyan-400/10 px-3 py-1.5 text-[10px] font-bold uppercase tracking-[.16em] text-cyan-300">
                <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-cyan-300" />{" "}
                Interactive computer vision laboratory
              </div>
              <h1 className="max-w-2xl text-[clamp(2.4rem,5vw,5.2rem)] font-black leading-[1.04] tracking-[-.055em]">
                See the road.
                <br />
                <span className="bg-gradient-to-r from-cyan-300 via-blue-400 to-violet-400 bg-clip-text text-transparent">
                  Beyond the surface.
                </span>
              </h1>
              <p className="mt-5 max-w-xl text-sm leading-7 text-slate-400 sm:text-base">
                An experimental road-inspection experience powered by a custom
                YOLO11s model. Analyze road images privately in your browser
                with ONNX Runtime Web.
              </p>
              <div className="mt-7 flex flex-wrap gap-2.5">
                <a
                  href="#experience"
                  className="inline-flex items-center gap-2 rounded-xl bg-cyan-400 px-5 py-3 text-sm font-bold text-[#041019] transition hover:bg-cyan-300"
                >
                  <UploadCloud size={18} /> Try live detection
                </a>
                <a
                  href="#showcase"
                  className="inline-flex items-center gap-2 rounded-xl border border-white/15 bg-white/5 px-5 py-3 text-sm font-semibold transition hover:bg-white/10"
                >
                  <Video size={18} /> Watch presentation
                </a>
              </div>
              <div className="mt-8 flex flex-wrap gap-2 text-[11px] font-semibold text-slate-400">
                <span className="rounded-full border border-white/10 px-3 py-1.5">
                  YOLO11s
                </span>
                <span className="rounded-full border border-white/10 px-3 py-1.5">
                  ONNX
                </span>
                <span className="rounded-full border border-white/10 px-3 py-1.5">
                  Next.js
                </span>
                <span className="rounded-full border border-white/10 px-3 py-1.5">
                  Browser inference
                </span>
              </div>
            </div>
            <div className="relative overflow-hidden rounded-2xl border border-cyan-400/20 bg-[#060f1e] p-4 shadow-[0_0_65px_rgba(34,211,238,.07)] sm:p-6">
              <div className="mb-5 flex items-center justify-between">
                <span className="text-xs font-bold uppercase tracking-[.2em] text-slate-400">
                  Vision / 04
                </span>
                <span className="flex items-center gap-1.5 text-[11px] text-emerald-300">
                  <span className="h-2 w-2 rounded-full bg-emerald-400" />{" "}
                  Detection classes
                </span>
              </div>
              <div className="relative aspect-[1.4] overflow-hidden rounded-xl bg-gradient-to-b from-[#102e47] via-[#1b3240] to-[#141d2c]">
                <svg
                  viewBox="0 0 500 340"
                  className="h-full w-full"
                  preserveAspectRatio="xMidYMid slice"
                  aria-label="Illustrated road damage inspection"
                  role="img"
                >
                  <path d="M0 150L180 125L220 160L0 230Z" fill="#153c3a" />
                  <path d="M500 150L330 125L290 160L500 230Z" fill="#123331" />
                  <path d="M200 130H300L500 340H0Z" fill="#414a55" />
                  <path
                    d="M250 145V175M250 205V235M250 270V320"
                    stroke="#d4e2e7"
                    strokeWidth="5"
                    strokeDasharray="24 15"
                  />
                  <path
                    d="M55 282L92 264L116 277L143 247L170 260L198 235"
                    fill="none"
                    stroke="#11151b"
                    strokeWidth="12"
                  />
                  <ellipse cx="365" cy="269" rx="55" ry="23" fill="#10141d" />
                  <ellipse cx="365" cy="269" rx="38" ry="12" fill="#273342" />
                </svg>
                <div className="absolute bottom-[12%] left-[9%] h-[32%] w-[37%] border-2 border-cyan-300 bg-cyan-400/10">
                  <span className="absolute -top-6 left-0 bg-cyan-300 px-2 py-1 text-[10px] font-black text-slate-950">
                    CRACK
                  </span>
                </div>
                <div className="absolute bottom-[10%] right-[9%] h-[32%] w-[30%] border-2 border-orange-400 bg-orange-400/10">
                  <span className="absolute -top-6 left-0 bg-orange-400 px-2 py-1 text-[10px] font-black text-slate-950">
                    POTHOLE
                  </span>
                </div>
                <div className="absolute left-3 top-3 rounded-lg border border-white/15 bg-[#07101d]/90 px-3 py-2 text-[10px] font-bold text-white">
                  ILLUSTRATIVE PREVIEW
                </div>
              </div>
              <div className="mt-4 grid grid-cols-4 gap-1.5 sm:gap-2">
                {classes.map((c) => (
                  <div
                    key={c.index}
                    className="min-w-0 rounded-lg border border-white/10 bg-white/[.04] p-2 text-center"
                  >
                    <div
                      className="mx-auto mb-2 h-1 w-6 rounded-full"
                      style={{ background: c.color }}
                    />
                    <p className="truncate text-[9px] font-semibold text-slate-300 sm:text-[11px]">
                      {c.name}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        <section className="mt-4 grid grid-cols-3 gap-2 sm:gap-4">
          {[
            { v: "2,804", l: "Dataset images" },
            { v: "44.97%", l: "Test mAP@50" },
            { v: "4", l: "Detection classes" },
          ].map((s) => (
            <div
              key={s.l}
              className="rounded-xl border border-white/10 bg-[#0b1424] px-3 py-4 text-center sm:py-5"
            >
              <p className="text-lg font-black text-cyan-300 sm:text-3xl">
                {s.v}
              </p>
              <p className="mt-1 text-[10px] text-slate-400 sm:text-xs">
                {s.l}
              </p>
            </div>
          ))}
        </section>

        <section
          id="experience"
          className="mt-5 scroll-mt-24 overflow-hidden rounded-[26px] border border-cyan-400/20 bg-[#0b1424] p-4 sm:p-7 lg:p-9"
        >
          <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="text-[10px] font-bold uppercase tracking-[.2em] text-cyan-400">
                01 / Interactive lab
              </p>
              <h2 className="mt-1 text-2xl font-black tracking-tight sm:text-3xl">
                Run the actual AI model
              </h2>
            </div>
            <span
              className={`rounded-full border px-3 py-2 text-[11px] font-bold ${modelState === "ready" ? "border-emerald-400/30 bg-emerald-400/10 text-emerald-300" : "border-white/10 bg-white/5 text-slate-300"}`}
            >
              {modelState === "ready"
                ? "● Model ready"
                : modelState === "loading"
                  ? "◌ Loading model"
                  : "! Model unavailable"}
            </span>
          </div>
          <div className="mb-4 flex gap-2 lg:hidden">
            <button
              onClick={() => setActiveTab("detect")}
              className={`flex-1 rounded-xl px-4 py-3 text-sm font-bold ${activeTab === "detect" ? "bg-cyan-400 text-[#05111a]" : "bg-white/5 text-slate-400"}`}
            >
              Upload image
            </button>
            <button
              onClick={() => setActiveTab("results")}
              className={`flex-1 rounded-xl px-4 py-3 text-sm font-bold ${activeTab === "results" ? "bg-cyan-400 text-[#05111a]" : "bg-white/5 text-slate-400"}`}
            >
              Results {result ? `(${result.total_detections})` : ""}
            </button>
          </div>
          <div className="grid gap-4 lg:grid-cols-2">
            <div
              className={`${activeTab === "results" ? "hidden lg:block" : ""} rounded-2xl border border-white/10 bg-[#07101d] p-4 sm:p-6`}
            >
              <label
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  chooseFile(e.dataTransfer.files[0]);
                  setActiveTab("detect");
                }}
                className="flex min-h-40 cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed border-cyan-400/35 bg-cyan-400/[.035] px-4 py-6 text-center transition hover:bg-cyan-400/10 sm:min-h-52"
              >
                <UploadCloud size={29} className="mb-3 text-cyan-300" />
                <span className="max-w-full break-all text-sm font-bold">
                  {file ? file.name : "Drop or choose a road image"}
                </span>
                <span className="mt-2 text-xs text-slate-500">
                  JPG / PNG / WEBP · up to 10 MB
                </span>
                <input
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  className="hidden"
                  onChange={(e) => chooseFile(e.target.files?.[0])}
                />
              </label>
              {preview && (
                <img
                  src={preview}
                  alt="Road image preview"
                  className="mt-3 max-h-56 w-full rounded-xl object-contain"
                />
              )}
              <button
                onClick={async () => {
                  await analyze();
                  setActiveTab("results");
                }}
                disabled={!file || loading || modelState === "error"}
                className="mt-3 flex w-full items-center justify-center gap-2 rounded-xl bg-cyan-400 px-5 py-3.5 text-sm font-black text-[#05111a] transition hover:bg-cyan-300 disabled:cursor-not-allowed disabled:opacity-40"
              >
                {loading ? (
                  <LoaderCircle size={18} className="animate-spin" />
                ) : (
                  <ScanSearch size={18} />
                )}{" "}
                {loading ? "Analyzing..." : "Analyze road image"}
              </button>
              <p className="mt-3 text-xs text-slate-500">
                {loading ? predictionProgress : modelProgress}
              </p>
              {modelState === "error" && (
                <button
                  onClick={() => void initializeModel().catch(() => undefined)}
                  className="mt-3 rounded-lg border border-rose-400/40 px-3 py-2 text-xs text-rose-300"
                >
                  Retry model
                </button>
              )}
              {error && (
                <p
                  role="alert"
                  className="mt-3 rounded-lg bg-rose-500/10 p-3 text-xs text-rose-300"
                >
                  {error}
                </p>
              )}
            </div>
            <div
              className={`${activeTab === "detect" ? "hidden lg:block" : ""} rounded-2xl border border-white/10 bg-[#07101d] p-4 sm:p-6`}
            >
              <div className="mb-3 flex items-center justify-between">
                <h3 className="text-sm font-bold">Detection output</h3>
                <span className="text-xs text-slate-500">
                  {inferenceTime !== null
                    ? `${inferenceTime.toFixed(0)} ms`
                    : "Awaiting analysis"}
                </span>
              </div>
              {!result ? (
                <div className="flex min-h-56 flex-col items-center justify-center rounded-xl border border-dashed border-white/10 text-center text-slate-500">
                  <ScanSearch size={35} className="mb-3 opacity-50" />
                  <p className="text-sm">
                    Your analyzed image will appear here
                  </p>
                  <button
                    onClick={() => setActiveTab("detect")}
                    className="mt-3 text-xs font-bold text-cyan-300 lg:hidden"
                  >
                    Choose an image
                  </button>
                </div>
              ) : (
                <>
                  <img
                    src={result.annotated_image}
                    alt="Annotated road detections"
                    className="max-h-72 w-full rounded-xl object-contain"
                  />
                  <div className="mt-3 flex items-center justify-between text-sm">
                    <b>{result.total_detections} detections</b>
                    <span className="text-slate-500">YOLO11s</span>
                  </div>
                  <div className="mt-3 max-h-28 space-y-1.5 overflow-y-auto">
                    {result.detections.map((d, i) => (
                      <div
                        key={i}
                        className="flex justify-between rounded-lg bg-white/5 px-3 py-2 text-xs"
                      >
                        <span className="capitalize">
                          {d.class_name.replaceAll("_", " ")}
                        </span>
                        <b className="text-cyan-300">
                          {(d.confidence * 100).toFixed(1)}%
                        </b>
                      </div>
                    ))}
                  </div>
                  <div className="mt-3 grid grid-cols-2 gap-2">
                    <button
                      onClick={downloadAnnotatedImage}
                      className="flex items-center justify-center gap-2 rounded-xl bg-white/10 px-3 py-3 text-xs font-bold"
                    >
                      <Download size={15} /> Image
                    </button>
                    <button
                      onClick={downloadReport}
                      className="flex items-center justify-center gap-2 rounded-xl bg-white/10 px-3 py-3 text-xs font-bold"
                    >
                      <Download size={15} /> Report
                    </button>
                  </div>
                </>
              )}
            </div>
          </div>
          <p className="mt-4 text-[11px] leading-5 text-slate-500">
            Academic proof of concept. Predictions can be inaccurate and require
            human verification. Image inference runs locally in your browser.
          </p>
        </section>

        <section
          id="showcase"
          className="mt-5 scroll-mt-24 overflow-hidden rounded-[26px] border border-white/10 bg-[#0b1424] p-4 sm:p-7 lg:p-9"
        >
          <div id="resources" className="scroll-mt-24">
            <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
              <div>
                <p className="text-[10px] font-bold uppercase tracking-[.2em] text-emerald-300">
                  02 / Research resources & presentation
                </p>
                <h2 className="mt-1 text-2xl font-black tracking-tight sm:text-3xl">
                  Explore the project.
                </h2>
                <p className="mt-1 text-xs text-slate-400 sm:text-sm">
                  Six project resources, followed by the original YouTube presentation.
                </p>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
              {resources.map((r) => (
                <button
                  key={r.name}
                  type="button"
                  onClick={() => startResourceDownload(r.url, r.name)}
                  aria-label={`Download ${r.name}`}
                  className="group flex min-w-0 flex-col items-start gap-2 rounded-xl border border-white/10 bg-[#0a1828] p-3 text-left transition duration-200 hover:-translate-y-0.5 hover:border-cyan-400/40 hover:bg-[#0d2135] focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-300"
                >
                  <div className="flex w-full items-center justify-between gap-1">
                    <span className="[&>span]:!h-9 [&>span]:!w-9 [&_img]:!h-5 [&_img]:!w-5">
                      <BrandLogo brand={r.logo} />
                    </span>
                    <Download size={15} className="shrink-0 text-cyan-300" />
                  </div>
                  <span className="text-xs font-bold leading-4 text-white sm:text-sm">{r.name}</span>
                  <span className="text-[10px] leading-4 text-slate-500">{r.detail}</span>
                </button>
              ))}
            </div>
            {downloadNotice && (
              <p role="status" className="mt-3 rounded-lg border border-cyan-400/20 bg-cyan-400/5 px-3 py-2 text-xs text-cyan-200">
                {downloadNotice}
              </p>
            )}
          </div>

          <div className="my-5 h-px bg-gradient-to-r from-transparent via-white/15 to-transparent sm:my-7" />

          <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="text-[10px] font-bold uppercase tracking-[.2em] text-violet-300">
                03 / Video walkthrough
              </p>
              <h3 className="mt-1 text-lg font-black sm:text-2xl">
                RoadVision AI — Project presentation
              </h3>
              <p className="mt-1 text-xs text-slate-400">
                Original YouTube player with the video's actual thumbnail.
              </p>
            </div>
            <a
              href={YOUTUBE_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 rounded-xl border border-white/15 px-3 py-2 text-xs font-bold transition hover:bg-white/10"
            >
              <Video size={16} className="text-red-400" /> Watch on YouTube <ExternalLink size={13} />
            </a>
          </div>
          <div className="relative aspect-video w-full overflow-hidden rounded-xl border border-white/10 bg-black sm:rounded-2xl">
            <iframe
              className="absolute inset-0 h-full w-full"
              src="https://www.youtube.com/embed/thgqco_hMpY?rel=0"
              title="RoadVision AI — Original YouTube project presentation"
              loading="lazy"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
              referrerPolicy="strict-origin-when-cross-origin"
              allowFullScreen
            />
          </div>
          <p className="mt-3 text-[11px] leading-5 text-slate-500">
            External downloads may require Google Drive confirmation or access permission.
            Video playback requires YouTube embedding to be enabled and a working connection.
          </p>
        </section>

        <footer className="mt-5 flex flex-wrap items-center justify-between gap-2 border-t border-white/10 pt-5 text-[11px] text-slate-500">
          <span>© RoadVision AI · Academic research prototype</span>
          <span>Built with YOLO11s · ONNX Runtime Web · Next.js</span>
        </footer>
      </div>
    </main>
  );
}
