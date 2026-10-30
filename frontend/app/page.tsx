

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
  ChartNoAxesCombined

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
    <main className="relative mx-auto max-w-[1500px] space-y-6 overflow-hidden px-4 pb-12 pt-5 text-slate-100 sm:px-6 lg:px-10">
      <div aria-hidden="true" className="pointer-events-none fixed inset-0 -z-10 bg-[#03071b]" />
      <div aria-hidden="true" className="pointer-events-none absolute left-1/3 top-0 -z-10 h-[750px] w-[750px] rounded-full bg-blue-700/20 blur-[150px]" />
      <header className="relative z-20 flex flex-wrap items-center justify-between gap-4 rounded-[22px] border border-indigo-400/20 bg-[#080d2b]/90 px-5 py-4 shadow-[0_16px_70px_rgba(0,0,0,.4)] backdrop-blur-xl lg:px-8">
        <a href="#top" className="flex items-center gap-3" aria-label="RoadVision AI home"><span className="flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-cyan-400 via-blue-600 to-violet-700 text-white shadow-[0_0_30px_rgba(45,110,255,.35)]"><BrainCircuit size={26}/></span><span><span className="block text-xl font-black tracking-tight">RoadVision <span className="bg-gradient-to-r from-cyan-300 to-blue-500 bg-clip-text text-transparent">AI</span></span><span className="text-[10px] tracking-[.18em] text-blue-200/65">INTELLIGENT ROAD INSPECTION</span></span></a>
        <nav className="hidden items-center gap-7 rounded-full border border-indigo-400/10 bg-[#090f32] px-7 py-3 text-sm text-blue-100/75 lg:flex"><a href="#top" className="text-white hover:text-cyan-300">Home</a><a href="#academic" className="hover:text-cyan-300">About</a><a href="#detection" className="hover:text-cyan-300">Detection</a><a href="#resources" className="hover:text-cyan-300">Resources</a></nav>
        <div className="flex items-center gap-3"><span role="status" className="hidden items-center gap-2 text-xs font-semibold text-emerald-300 sm:inline-flex">{modelState === "ready" ? <CheckCircle2 size={15}/> : modelState === "loading" ? <LoaderCircle size={15} className="animate-spin"/> : <AlertTriangle size={15} className="text-rose-400"/>}{modelState === "ready" ? "Model Ready" : modelState === "loading" ? "Loading Model" : "Model Unavailable"}</span><a href="#detection" className="inline-flex items-center gap-2 rounded-full bg-gradient-to-r from-blue-600 to-violet-600 px-5 py-3 text-xs font-bold text-white shadow-[0_8px_32px_rgba(66,65,255,.35)] transition hover:brightness-125 sm:text-sm">Try AI Detection <ArrowRight size={16}/></a></div>
      </header>

      <section id="top" className="relative isolate overflow-hidden rounded-[30px] border border-blue-500/20 bg-[#080d2b] px-5 pb-8 pt-12 shadow-[0_24px_100px_rgba(0,0,0,.5)] sm:px-8 lg:px-12 lg:pb-14 lg:pt-16">
        <div aria-hidden="true" className="pointer-events-none absolute inset-0 opacity-30" style={{backgroundImage:"linear-gradient(rgba(91,103,238,.12) 1px,transparent 1px),linear-gradient(90deg,rgba(91,103,238,.12) 1px,transparent 1px)",backgroundSize:"76px 76px",maskImage:"linear-gradient(to bottom,black,transparent 85%)"}}/>
        <div aria-hidden="true" className="pointer-events-none absolute -left-32 top-32 h-[450px] w-[260px] rotate-[-24deg] rounded-[110px] border border-blue-400/60 bg-gradient-to-b from-blue-950 via-blue-700 to-indigo-700 opacity-65 blur-[1px] shadow-[0_0_80px_rgba(44,88,255,.65)]"/>
        <div aria-hidden="true" className="pointer-events-none absolute -right-32 top-16 h-[440px] w-[260px] rotate-[24deg] rounded-[110px] border border-cyan-400/50 bg-gradient-to-b from-indigo-950 via-blue-700 to-violet-700 opacity-65 shadow-[0_0_80px_rgba(44,88,255,.6)]"/>
        <div className="relative z-10 mx-auto max-w-4xl text-center"><span className="inline-flex items-center gap-2 rounded-full border border-blue-400/25 bg-blue-600/10 px-4 py-2 text-[11px] font-bold uppercase tracking-[.2em] text-cyan-200"><Sparkles size={14}/> AI Powered Road Inspection</span><h1 className="mt-7 text-4xl font-black leading-[1.08] tracking-tight text-white sm:text-5xl lg:text-[70px]">Smarter Roads with<br/><span className="bg-gradient-to-r from-cyan-300 via-blue-500 to-violet-400 bg-clip-text text-transparent">AI-Powered Detection</span></h1><p className="mx-auto mt-6 max-w-2xl text-sm leading-7 text-blue-100/65 sm:text-base">Detect cracks, potholes, surface damage and manholes using a custom-trained YOLO11s model — directly in your browser, with no image upload to a server.</p><div className="mt-8 flex flex-wrap items-center justify-center gap-3"><a href="#detection" className="inline-flex items-center gap-2 rounded-full bg-gradient-to-r from-blue-600 to-violet-600 px-7 py-3.5 text-sm font-bold shadow-[0_0_32px_rgba(62,83,255,.45)] transition hover:scale-[1.03]"><UploadCloud size={18}/> Try AI Detection <ArrowRight size={17}/></a><a href="#resources" className="inline-flex items-center gap-2 rounded-full border border-indigo-400/30 bg-[#11183c]/80 px-7 py-3.5 text-sm font-semibold text-blue-100 transition hover:border-cyan-400/60">Explore Resources <ExternalLink size={16}/></a></div></div>
        <div className="relative z-10 mx-auto mt-12 max-w-5xl rounded-[26px] border border-blue-400/40 bg-[#070d27]/95 p-3 shadow-[0_0_80px_rgba(27,81,255,.27)] sm:p-5"><div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/10 px-2 pb-4"><div className="flex items-center gap-2 text-xs font-semibold text-blue-100"><ScanSearch size={17} className="text-cyan-400"/> AI Detection Preview <span className="rounded-full bg-emerald-400/10 px-2 py-1 text-[10px] text-emerald-300">Illustrative UI</span></div><span className="text-[11px] text-blue-200/60">YOLO11s · ONNX Runtime Web</span></div><div className="grid gap-4 pt-4 md:grid-cols-[1.8fr_1fr]"><div className="relative min-h-[240px] overflow-hidden rounded-xl border border-blue-500/20 bg-[#0b1634] sm:min-h-[320px]"><svg viewBox="0 0 680 360" preserveAspectRatio="xMidYMid slice" className="absolute inset-0 h-full w-full" role="img" aria-label="Illustration of a road inspection with sample detection boxes"><defs><linearGradient id="roadsky" x1="0" x2="0" y1="0" y2="1"><stop stopColor="#1a335d"/><stop offset="1" stopColor="#10192f"/></linearGradient><linearGradient id="roadpavement" x1="0" x2="0" y1="0" y2="1"><stop stopColor="#41465b"/><stop offset="1" stopColor="#1a2035"/></linearGradient></defs><rect width="680" height="360" fill="url(#roadsky)"/><path d="M0 125 L240 108 L270 180 L0 250Z" fill="#142b2d"/><path d="M680 115 L430 100 L405 180 L680 235Z" fill="#142b2d"/><path d="M270 130 L412 130 L680 360 L0 360Z" fill="url(#roadpavement)"/><path d="M340 146 L340 172 M340 195 L340 237 M340 265 L340 330" stroke="#d7dbe7" strokeWidth="5" strokeDasharray="26 18"/><path d="M125 298 l30 -17 18 14 25 -25 20 15 30 -18" stroke="#0a0e1c" strokeWidth="9" fill="none"/><ellipse cx="474" cy="277" rx="61" ry="22" fill="#0b101d"/><ellipse cx="474" cy="277" rx="45" ry="13" fill="#323b50"/><path d="M410 220 l18 -14 24 10 18 -16 17 12" stroke="#0b101d" strokeWidth="10" fill="none"/></svg><div className="absolute left-[12%] top-[54%] h-[24%] w-[29%] border-2 border-cyan-400 bg-cyan-400/10"><span className="absolute -top-6 left-0 whitespace-nowrap rounded-t bg-cyan-500 px-2 py-1 text-[10px] font-bold text-slate-950">Crack · Sample</span></div><div className="absolute right-[14%] top-[61%] h-[23%] w-[29%] border-2 border-violet-400 bg-violet-400/10"><span className="absolute -top-6 left-0 whitespace-nowrap rounded-t bg-violet-500 px-2 py-1 text-[10px] font-bold text-white">Pothole · Sample</span></div><div className="absolute bottom-3 left-3 rounded-lg border border-emerald-400/20 bg-[#07152a]/90 px-3 py-2 text-[10px] text-emerald-300">● Browser inference enabled</div></div><div className="rounded-xl border border-indigo-400/20 bg-[#0a1230] p-4"><div className="flex items-center justify-between"><span className="text-sm font-bold">Supported Classes</span><span className="rounded-full bg-violet-600 px-2.5 py-1 text-xs font-bold">04</span></div><div className="mt-5 space-y-3">{CLASS_NAMES.map((name,i)=><div key={name} className="flex items-center justify-between rounded-xl border border-white/5 bg-white/[.035] px-3 py-3 text-xs"><span className="flex items-center gap-2 capitalize"><span className="h-2.5 w-2.5 rounded" style={{background:CLASS_COLORS[i]}}/>{name.replaceAll("_"," ")}</span><CheckCircle2 size={15} className="text-blue-300/60"/></div>)}</div><p className="mt-5 text-[11px] leading-5 text-blue-200/50">This hero preview is illustrative. Upload a road image below for real model predictions.</p></div></div></div>
        <div className="relative z-10 mt-8 flex flex-wrap justify-center gap-5 text-xs text-blue-100/70 sm:gap-10"><span className="flex items-center gap-2"><LockKeyhole size={16} className="text-cyan-400"/> Private Browser Processing</span><span className="flex items-center gap-2"><Cpu size={16} className="text-blue-400"/> ONNX Inference</span><span className="flex items-center gap-2"><Layers3 size={16} className="text-violet-400"/> 4 Detection Classes</span></div>
      </section>
      <section aria-label="Model performance" className="grid grid-cols-2 gap-3 md:grid-cols-4 md:gap-4">{[{label:"Model",value:"YOLO11s",note:"ONNX inference",icon:Cpu},{label:"Test mAP50",value:"44.97%",note:"Measured evaluation",icon:ChartNoAxesCombined},{label:"Damage Classes",value:"04",note:"Road categories",icon:Layers3},{label:"Test Recall",value:"49.35%",note:"Measured evaluation",icon:ScanSearch}].map(stat=>{const Icon=stat.icon;return <div key={stat.label} className="rounded-2xl border border-blue-500/20 bg-gradient-to-br from-[#11183f] to-[#080e29] p-5 shadow-[0_12px_35px_rgba(0,0,0,.22)] transition hover:-translate-y-1 hover:border-blue-400/60"><div className="mb-3 flex items-center justify-between"><Icon size={23} className="text-blue-400"/><span className="h-1.5 w-1.5 rounded-full bg-cyan-400 shadow-[0_0_12px_#22d3ee]"/></div><p className="text-2xl font-black text-white sm:text-3xl">{stat.value}</p><p className="mt-2 text-xs font-semibold text-blue-100">{stat.label}</p><p className="mt-1 text-[11px] text-blue-200/50">{stat.note}</p></div>})}</section>
      <section id="academic" aria-labelledby="academic-details" className="overflow-hidden rounded-2xl border border-blue-500/20 bg-[#0a1030] shadow-xl"><div className="flex flex-wrap items-center justify-between gap-3 border-b border-blue-400/10 bg-gradient-to-r from-indigo-700/20 to-transparent px-5 py-5 sm:px-7"><div className="flex items-center gap-3"><div className="flex h-11 w-11 items-center justify-center rounded-xl bg-blue-500/15 text-cyan-300"><GraduationCap size={23}/></div><div><p className="text-[11px] font-bold uppercase tracking-[.18em] text-cyan-300">Academic Submission</p><h2 id="academic-details" className="mt-1 text-xl font-bold sm:text-2xl">Project & Student Details</h2></div></div><span className="rounded-full border border-blue-400/20 bg-blue-500/10 px-3 py-1.5 text-xs font-semibold text-blue-200">Superior University</span></div><dl className="grid gap-px bg-blue-500/10 sm:grid-cols-2 lg:grid-cols-3">{[["Student Name","Naveed Ahmed"],["Roll Number","SU92-BSCSM-F23-600"],["Class","BSCS (7A)"],["Subject","Deep Learning"],["Instructor","Dr. Munir Ahmad"],["University","Superior University"]].map(([label,value])=><div key={label} className="bg-[#0b1232] px-5 py-5 sm:px-7"><dt className="text-xs uppercase tracking-wider text-blue-200/50">{label}</dt><dd className="mt-2 break-words text-sm font-bold text-white sm:text-base">{value}</dd></div>)}</dl></section>
      <section className="space-y-3 rounded-2xl border border-blue-500/20 bg-[#0b1232] p-5 shadow-xl sm:p-7">

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

            className="rounded-xl bg-gradient-to-r from-blue-600 to-violet-600 px-5 py-3 font-semibold text-white"

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

      <section id="detection" className="grid scroll-mt-6 gap-6 lg:grid-cols-2">

        <div className="space-y-5 rounded-2xl border border-blue-500/20 bg-[#0b1232] p-5 shadow-xl sm:p-7">

          <h3 className="text-xl font-bold flex gap-2 items-center">

            <UploadCloud className="text-cyan-400" />

            AI Damage Detection

          </h3>

          <label

            className="group block cursor-pointer rounded-2xl border-2 border-dashed border-cyan-400/25 bg-gradient-to-b from-cyan-400/5 to-transparent p-8 text-center transition-all hover:border-cyan-400 hover:bg-cyan-400/10"

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

            className="rounded-xl bg-gradient-to-r from-blue-600 to-violet-600 px-5 py-3 font-semibold text-white transition hover:brightness-125 disabled:opacity-50 w-full flex items-center justify-center gap-2"

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

        <div className="space-y-5 rounded-2xl border border-blue-500/20 bg-[#0b1232] p-5 shadow-xl sm:p-7">

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

                  className="rounded-xl bg-gradient-to-r from-blue-600 to-violet-600 px-5 py-3 font-semibold text-white transition hover:brightness-125 disabled:opacity-50 flex items-center justify-center gap-2"

                  onClick={downloadReport}

                >

                  <Download size={18} />

                  Report

                </button>

                <button

                  className="rounded-xl bg-gradient-to-r from-blue-600 to-violet-600 px-5 py-3 font-semibold text-white transition hover:brightness-125 disabled:opacity-50 flex items-center justify-center gap-2"

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

      <section className="rounded-2xl border border-blue-500/20 bg-[#0b1232] p-5 shadow-xl sm:p-7">

        <h3 className="text-xl font-bold mb-5">

          Supported Detection Classes

        </h3>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">

          {CLASS_NAMES.map(

            (name, i) => (

              <div

                key={name}

                className="rounded-2xl border border-white/5 bg-slate-800/70 p-5 text-center transition hover:border-cyan-400/30 hover:bg-slate-800"

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

      <section id="resources" className="panel scroll-mt-6">

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

                className="resource rounded-xl border border-blue-500/20 bg-blue-500/10 px-4 py-3 text-sm font-semibold text-blue-100 transition hover:bg-blue-500/20 flex items-center justify-center gap-2 rounded-xl border border-white/10 transition hover:border-cyan-400/40 hover:bg-cyan-400/10"

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

                className="resource rounded-xl border border-blue-500/20 bg-blue-500/10 px-4 py-3 text-sm font-semibold text-blue-100 transition hover:bg-blue-500/20 flex items-center justify-center gap-2 rounded-xl border border-white/5 opacity-50"

              >

                <Icon size={18} />

                {resource.name} — Coming Soon

              </div>

            );

          })}

        </div>

      </section>

      <footer className="border-t border-white/10 py-7 text-center text-xs leading-6 text-slate-500 sm:text-sm">

        RoadVision AI · Naveed Ahmed · SU92-BSCSM-F23-600 · BSCS (7A) · Deep Learning · Dr. Munir Ahmad · Superior University · Predictions require human verification

      </footer>

    </main>

  );

}
