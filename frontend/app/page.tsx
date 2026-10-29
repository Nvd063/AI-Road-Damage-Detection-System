
"use client";

import { useEffect, useState } from "react";
import {
  Activity, UploadCloud, Download, Github,
  FileText, Database, Video, BrainCircuit,
  AlertTriangle, ExternalLink
} from "lucide-react";

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

const API = process.env.NEXT_PUBLIC_API_URL || "";

const resources = [
  { name: "GitHub Repository", icon: Github, url: process.env.NEXT_PUBLIC_GITHUB_URL },
  { name: "Dataset", icon: Database, url: process.env.NEXT_PUBLIC_DATASET_URL },
  { name: "Model Weights", icon: BrainCircuit, url: process.env.NEXT_PUBLIC_WEIGHTS_URL },
  { name: "Project Report", icon: FileText, url: process.env.NEXT_PUBLIC_REPORT_URL },
  { name: "Presentation", icon: FileText, url: process.env.NEXT_PUBLIC_PRESENTATION_URL },
  { name: "YouTube Demo", icon: Video, url: process.env.NEXT_PUBLIC_YOUTUBE_URL }
];

const classes = ["Crack", "Pothole", "Surface Damage", "Manhole"];

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [apiStatus, setApiStatus] = useState("Checking");

  useEffect(() => {
    if (!API) {
      setApiStatus("Not configured");
      return;
    }
    fetch(`${API}/health`)
      .then(r => {
        if (!r.ok) throw new Error();
        setApiStatus("Online");
      })
      .catch(() => setApiStatus("Unavailable"));
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

  function chooseFile(f?: File) {
    setResult(null);
    setError("");
    if (!f) return;
    if (!["image/jpeg", "image/png", "image/webp"].includes(f.type)) {
      setError("Only JPG, PNG or WEBP images are allowed.");
      return;
    }
    if (f.size > 10 * 1024 * 1024) {
      setError("Maximum image size is 10 MB.");
      return;
    }
    setFile(f);
  }

  async function analyze() {
    if (!file || !API) {
      setError("Select an image and configure the backend API.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const data = new FormData();
      data.append("file", file);

      const response = await fetch(`${API}/predict`, {
        method: "POST",
        body: data
      });

      if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        throw new Error(detail.detail || "Prediction failed");
      }

      setResult(await response.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Connection failed");
    } finally {
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
      `Total detections: ${result.total_detections}`,
      "",
      ...result.detections.map((d, i) =>
        `${i + 1}. ${d.class_name} | Confidence: ${(d.confidence * 100).toFixed(1)}% | Box: ${d.bbox.join(", ")}`
      ),
      "",
      "AI-generated results require human verification."
    ].join("\n");

    const blob = new Blob([report], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "road-damage-report.txt";
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <main className="max-w-7xl mx-auto px-5 py-8 space-y-8">
      <header className="flex flex-wrap justify-between items-center gap-4 border-b border-slate-800 pb-6">
        <div className="flex items-center gap-3">
          <BrainCircuit className="text-cyan-400" size={32}/>
          <div>
            <h1 className="text-2xl font-bold">RoadVision AI</h1>
            <p className="text-slate-400 text-sm">Automated Road Damage Detection & Reporting</p>
          </div>
        </div>
        <div className="text-sm bg-slate-800 rounded-full px-4 py-2">
          API: <span className="text-cyan-400">{apiStatus}</span>
        </div>
      </header>

      <section className="py-7">
        <p className="text-cyan-400 font-semibold mb-3">DEEP LEARNING PROJECT</p>
        <h2 className="text-4xl md:text-5xl font-bold max-w-3xl leading-tight">
          Smarter Road Inspection Powered by Artificial Intelligence
        </h2>
        <p className="text-slate-400 mt-5 max-w-2xl">
          Detect cracks, potholes, surface damage and manholes using a custom-trained YOLO11s model.
        </p>
      </section>

      <section className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          ["Model", "YOLO11s"],
          ["Damage Classes", "4"],
          ["Test mAP50", "44.97%"],
          ["Test Recall", "49.35%"]
        ].map(([label, value]) => (
          <div className="panel" key={label}>
            <p className="text-slate-400 text-sm">{label}</p>
            <p className="text-2xl font-bold mt-2">{value}</p>
          </div>
        ))}
      </section>

      <section className="grid lg:grid-cols-2 gap-6">
        <div className="panel space-y-5">
          <h3 className="text-xl font-bold flex gap-2 items-center">
            <UploadCloud className="text-cyan-400"/> AI Damage Detection
          </h3>

          <label
            className="border-2 border-dashed border-slate-600 rounded-xl p-8 text-center block cursor-pointer hover:border-cyan-400"
            onDragOver={e => e.preventDefault()}
            onDrop={e => {
              e.preventDefault();
              chooseFile(e.dataTransfer.files[0]);
            }}
          >
            <UploadCloud className="mx-auto text-cyan-400 mb-3" size={36}/>
            <p>{file ? file.name : "Drop an image here or click to browse"}</p>
            <p className="text-slate-500 text-sm mt-2">JPG, PNG, WEBP · Max 10 MB</p>
            <input
              type="file"
              accept="image/jpeg,image/png,image/webp"
              className="hidden"
              onChange={e => chooseFile(e.target.files?.[0])}
            />
          </label>

          {preview && (
            <img src={preview} alt="Uploaded road" className="rounded-xl w-full"/>
          )}

          <button
            className="btn w-full"
            disabled={!file || loading || !API}
            onClick={analyze}
          >
            {loading ? "Analyzing Image..." : "Run AI Detection"}
          </button>

          {error && <p className="text-red-400 text-sm">{error}</p>}
        </div>

        <div className="panel space-y-5">
          <h3 className="text-xl font-bold flex gap-2 items-center">
            <Activity className="text-cyan-400"/> Detection Results
          </h3>

          {!result ? (
            <div className="min-h-64 flex items-center justify-center text-slate-500 text-center">
              Upload an image and run detection to view AI predictions.
            </div>
          ) : (
            <>
              <img
                src={result.annotated_image}
                alt="AI annotated detection"
                className="w-full rounded-xl"
              />
              <p className="font-semibold">
                {result.total_detections} detections found
              </p>
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {result.detections.map((d, i) => (
                  <div key={i} className="flex justify-between gap-3 bg-slate-800 rounded-lg p-3 text-sm">
                    <span className="capitalize">{d.class_name.replaceAll("_", " ")}</span>
                    <span className="text-cyan-400">{(d.confidence * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
              <button className="btn w-full flex items-center justify-center gap-2" onClick={downloadReport}>
                <Download size={18}/> Download Detection Report
              </button>
            </>
          )}
        </div>
      </section>

      <section className="panel">
        <h3 className="text-xl font-bold mb-5">Supported Detection Classes</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {classes.map(c => (
            <div key={c} className="bg-slate-800 rounded-xl p-5 text-center">
              <AlertTriangle className="mx-auto mb-3 text-cyan-400"/>
              <p className="font-semibold">{c}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="panel">
        <h3 className="text-xl font-bold mb-5">Project Resources</h3>
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {resources.map(r => {
            const Icon = r.icon;
            return r.url ? (
              <a
                key={r.name}
                className="resource flex justify-center items-center gap-2"
                href={r.url}
                target="_blank"
                rel="noopener noreferrer"
              >
                <Icon size={18}/>{r.name}<ExternalLink size={14}/>
              </a>
            ) : (
              <div key={r.name} className="resource opacity-50 flex justify-center items-center gap-2">
                <Icon size={18}/>{r.name} — Coming Soon
              </div>
            );
          })}
        </div>
      </section>

      <footer className="text-center text-slate-500 text-sm py-6">
        RoadVision AI · Academic Deep Learning Project · Predictions require human verification
      </footer>
    </main>
  );
}
