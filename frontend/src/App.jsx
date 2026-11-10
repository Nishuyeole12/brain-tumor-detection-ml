
import { useEffect, useRef, useState } from "react";
import {
  Activity,
  Brain,
  ChartNoAxesCombined,
  CheckCircle2,
  ChevronRight,
  CircleAlert,
  Clock3,
  FileImage,
  HeartPulse,
  History,
  ImagePlus,
  LayoutDashboard,
  LoaderCircle,
  Menu,
  RotateCcw,
  ScanSearch,
  ShieldCheck,
  Stethoscope,
  Upload,
  Wifi,
  WifiOff,
  X,
} from "lucide-react";

import "./App.css";

const API_URL = "http://127.0.0.1:8000";

const CLASS_NAMES = [
  "Glioma Tumor",
  "Meningioma Tumor",
  "No Tumor",
  "Pituitary Tumor",
];

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [backendOnline, setBackendOnline] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [history, setHistory] = useState([]);
  const [mobileMenu, setMobileMenu] = useState(false);

  const inputRef = useRef(null);

  // Check backend connection
  useEffect(() => {
    let active = true;

    const checkBackend = async () => {
      try {
        const response = await fetch(`${API_URL}/`);
        if (active) setBackendOnline(response.ok);
      } catch {
        if (active) setBackendOnline(false);
      }
    };

    checkBackend();
    const interval = setInterval(checkBackend, 10000);

    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  // Clean up image preview URL
  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  // Select image
  const selectImage = (selectedFile) => {
    setError("");
    setResult(null);

    if (!selectedFile) return;

    if (!["image/jpeg", "image/png"].includes(selectedFile.type)) {
      setError("Please select a JPG or PNG image.");
      return;
    }

    if (selectedFile.size > 10 * 1024 * 1024) {
      setError("Image size must be less than 10 MB.");
      return;
    }

    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
  };

  const handleFileChange = (event) => {
    selectImage(event.target.files?.[0]);
    event.target.value = "";
  };

  // Drag and drop
  const handleDrop = (event) => {
    event.preventDefault();
    setDragging(false);
    selectImage(event.dataTransfer.files?.[0]);
  };

  // Connect to FastAPI prediction endpoint
  const analyzeMRI = async () => {
    if (!file) {
      setError("Please upload an MRI image first.");
      return;
    }

    if (!backendOnline) {
      setError("Backend is offline. Start your FastAPI server first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Prediction request failed.");
      }

      if (!data.success || !CLASS_NAMES.includes(data.prediction)) {
        throw new Error("The backend returned an unexpected prediction.");
      }

      const prediction = {
        ...data,
        id: Date.now(),
        fileName: file.name,
        date: new Date().toLocaleString(),
      };

      setResult(prediction);
      setHistory((previous) => [prediction, ...previous].slice(0, 10));
    } catch (err) {
      setError(
        err.message === "Failed to fetch"
          ? "Cannot connect to backend. Check that FastAPI is running."
          : err.message
      );
    } finally {
      setLoading(false);
    }
  };

  // Reset current analysis
  const resetAnalysis = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError("");

    if (inputRef.current) {
      inputRef.current.value = "";
    }
  };

  const getResultStyle = (name) => {
    if (name === "No Tumor") return "healthy";
    return "detected";
  };

  return (
    <div className="app-layout">

      {/* SIDEBAR */}
      <aside className={`sidebar ${mobileMenu ? "sidebar-open" : ""}`}>

        <div className="brand">
          <div className="brand-icon">
            <Brain size={27} />
          </div>

          <div>
            <h2>BrainCare<span> AI</span></h2>
            <p>Medical Intelligence</p>
          </div>

          <button
            className="mobile-close"
            onClick={() => setMobileMenu(false)}
            aria-label="Close menu"
          >
            <X size={20} />
          </button>
        </div>

        <div className="sidebar-label">MAIN MENU</div>

        <nav className="sidebar-nav">
          <a className="nav-item active" href="#dashboard">
            <LayoutDashboard size={19} />
            Dashboard
          </a>

          <a className="nav-item" href="#scanner">
            <ScanSearch size={19} />
            MRI Analysis
          </a>

          <a className="nav-item" href="#results">
            <ChartNoAxesCombined size={19} />
            Results
          </a>

          <a className="nav-item" href="#history">
            <History size={19} />
            History
          </a>
        </nav>

        <div className="sidebar-bottom">
          <div className="support-card">
            <div className="support-icon">
              <HeartPulse size={22} />
            </div>

            <h3>AI Medical Assistant</h3>
            <p>
              AI-assisted MRI image classification for educational
              and research purposes.
            </p>
          </div>

          <div className="sidebar-footer">
            <ShieldCheck size={17} />
            <span>Privacy & Security</span>
          </div>
        </div>
      </aside>

      {mobileMenu && (
        <div
          className="mobile-overlay"
          onClick={() => setMobileMenu(false)}
        />
      )}

      {/* MAIN CONTENT */}
      <main className="main-content" id="dashboard">

        {/* TOP NAVIGATION */}
        <header className="topbar">
          <div className="topbar-left">
            <button
              className="menu-button"
              onClick={() => setMobileMenu(true)}
              aria-label="Open menu"
            >
              <Menu size={22} />
            </button>

            <div>
              <p className="breadcrumb">Workspace / Dashboard</p>
              <h2>Brain MRI Dashboard</h2>
            </div>
          </div>

          <div className="topbar-right">
            <div className={`connection-status ${backendOnline ? "online" : "offline"}`}>
              {backendOnline ? <Wifi size={15} /> : <WifiOff size={15} />}
              {backendOnline ? "Backend Connected" : "Backend Offline"}
            </div>

            <div className="profile-avatar">BC</div>
          </div>
        </header>

        <div className="dashboard-content">

          {/* WELCOME SECTION */}
          <section className="welcome-section">
            <div>
              <p className="section-eyebrow">AI-POWERED HEALTHCARE</p>
              <h1>Welcome to BrainCare AI</h1>
              <p className="welcome-description">
                Upload a brain MRI image and use your trained deep
                learning model for four-class image classification.
              </p>
            </div>

            <div className="welcome-illustration">
              <Brain size={55} strokeWidth={1.3} />
              <span>AI MRI Analysis</span>
            </div>
          </section>

          {/* OVERVIEW CARDS */}
          <section className="overview-grid">
            <div className="overview-card">
              <div className="overview-icon blue">
                <Brain size={22} />
              </div>
              <div>
                <p>AI Model</p>
                <h3>Efficient CNN</h3>
                <span>TensorFlow / Keras</span>
              </div>
            </div>

            <div className="overview-card">
              <div className="overview-icon purple">
                <ScanSearch size={22} />
              </div>
              <div>
                <p>Supported Classes</p>
                <h3>04 Classes</h3>
                <span>Four-class classification</span>
              </div>
            </div>

            <div className="overview-card">
              <div className="overview-icon green">
                <Activity size={22} />
              </div>
              <div>
                <p>Backend Status</p>
                <h3>{backendOnline ? "Connected" : "Offline"}</h3>
                <span>{backendOnline ? "FastAPI available" : "Check API server"}</span>
              </div>
            </div>
          </section>

          {/* ANALYSIS SECTION */}
          <section className="analysis-grid" id="scanner">

            {/* UPLOAD PANEL */}
            <div className="panel upload-panel">
              <div className="panel-heading">
                <div>
                  <h2>Upload MRI Scan</h2>
                  <p>Select an MRI image for AI analysis.</p>
                </div>

                <div className="panel-heading-icon">
                  <ImagePlus size={21} />
                </div>
              </div>

              <div
                className={`drop-zone ${dragging ? "dragging" : ""} ${preview ? "has-preview" : ""}`}
                onDragOver={(event) => {
                  event.preventDefault();
                  setDragging(true);
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={handleDrop}
                onClick={() => inputRef.current?.click()}
                role="button"
                tabIndex={0}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    inputRef.current?.click();
                  }
                }}
              >
                {preview ? (
                  <div className="preview-wrapper">
                    <img
                      src={preview}
                      alt="Uploaded brain MRI"
                      className="mri-preview"
                    />

                    <div className="preview-overlay">
                      <span>Image Preview</span>
                    </div>
                  </div>
                ) : (
                  <div className="upload-placeholder">
                    <div className="upload-circle">
                      <Upload size={29} />
                    </div>

                    <h3>Upload your MRI image</h3>
                    <p>Drag and drop your image here</p>
                    <span>or click to browse files</span>

                    <div className="file-types">
                      <span>JPG</span>
                      <span>JPEG</span>
                      <span>PNG</span>
                    </div>
                  </div>
                )}

                <input
                  ref={inputRef}
                  type="file"
                  accept="image/jpeg,image/png"
                  hidden
                  onChange={handleFileChange}
                />
              </div>

              {file && (
                <div className="selected-file">
                  <div className="file-icon">
                    <FileImage size={22} />
                  </div>

                  <div className="file-info">
                    <strong>{file.name}</strong>
                    <span>{(file.size / 1024 / 1024).toFixed(2)} MB</span>
                  </div>

                  <button
                    className="remove-file"
                    onClick={(event) => {
                      event.stopPropagation();
                      resetAnalysis();
                    }}
                    aria-label="Remove image"
                  >
                    <X size={19} />
                  </button>
                </div>
              )}

              <div className="upload-actions">
                <button
                  className="analyze-button"
                  onClick={analyzeMRI}
                  disabled={!file || loading || !backendOnline}
                >
                  {loading ? (
                    <>
                      <LoaderCircle className="spin" size={19} />
                      Analyzing MRI...
                    </>
                  ) : (
                    <>
                      <ScanSearch size={19} />
                      Analyze MRI
                      <ChevronRight size={18} />
                    </>
                  )}
                </button>

                <button
                  className="reset-button"
                  onClick={resetAnalysis}
                  disabled={!file && !result}
                >
                  <RotateCcw size={17} />
                  Reset
                </button>
              </div>

              {error && (
                <div className="error-message">
                  <CircleAlert size={18} />
                  <span>{error}</span>
                </div>
              )}

              <div className="upload-note">
                <ShieldCheck size={16} />
                Image is sent to your local backend for processing.
              </div>
            </div>

            {/* RESULT PANEL */}
            <div className="panel result-panel" id="results">
              <div className="panel-heading">
                <div>
                  <h2>Analysis Results</h2>
                  <p>AI prediction and confidence scores.</p>
                </div>

                <div className="panel-heading-icon">
                  <ChartNoAxesCombined size={21} />
                </div>
              </div>

              {!result ? (
                <div className="empty-result">
                  <div className="empty-result-icon">
                    <Brain size={42} strokeWidth={1.3} />
                  </div>

                  <h3>Awaiting MRI Analysis</h3>

                  <p>
                    Upload an MRI image and click Analyze MRI
                    to view the model prediction.
                  </p>

                  <div className="empty-result-footer">
                    <Clock3 size={16} />
                    No analysis performed yet
                  </div>
                </div>
              ) : (
                <div className="result-content">

                  <div className="result-success">
                    <CheckCircle2 size={19} />
                    Analysis completed successfully
                  </div>

                  <div className={`prediction-card ${getResultStyle(result.prediction)}`}>
                    <span className="prediction-label">
                      MODEL PREDICTION
                    </span>

                    <h3>{result.prediction}</h3>

                    <div className="confidence-display">
                      <span>Model confidence</span>
                      <strong>{Number(result.confidence).toFixed(2)}%</strong>
                    </div>

                    <div className="confidence-track">
                      <div
                        className="confidence-fill"
                        style={{
                          width: `${Math.min(Number(result.confidence), 100)}%`,
                        }}
                      />
                    </div>
                  </div>

                  <div className="probability-section">
                    <h3>Class-wise Probabilities</h3>

                    {CLASS_NAMES.map((name) => {
                      const score = Number(result.probabilities?.[name] ?? 0);

                      return (
                        <div className="probability-row" key={name}>
                          <div className="probability-label">
                            <span>{name}</span>
                            <strong>{score.toFixed(2)}%</strong>
                          </div>

                          <div className="probability-track">
                            <div
                              className="probability-fill"
                              style={{ width: `${Math.min(score, 100)}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>

                  <div className="result-file">
                    <FileImage size={17} />
                    <span>{result.fileName}</span>
                  </div>
                </div>
              )}
            </div>
          </section>

          {/* HISTORY SECTION */}
          <section className="panel history-panel" id="history">
            <div className="panel-heading">
              <div>
                <h2>Recent Analysis</h2>
                <p>Results from this browser session.</p>
              </div>

              <div className="panel-heading-icon">
                <History size={21} />
              </div>
            </div>

            {history.length === 0 ? (
              <div className="empty-history">
                <History size={25} />
                <span>No analysis history available yet.</span>
              </div>
            ) : (
              <div className="history-table-wrapper">
                <table className="history-table">
                  <thead>
                    <tr>
                      <th>Image</th>
                      <th>Prediction</th>
                      <th>Confidence</th>
                      <th>Date & Time</th>
                    </tr>
                  </thead>

                  <tbody>
                    {history.map((item) => (
                      <tr key={item.id}>
                        <td>{item.fileName}</td>
                        <td>
                          <span className={`history-badge ${getResultStyle(item.prediction)}`}>
                            {item.prediction}
                          </span>
                        </td>
                        <td>{Number(item.confidence).toFixed(2)}%</td>
                        <td>{item.date}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>

          {/* DISCLAIMER */}
          <footer className="disclaimer">
            <Stethoscope size={20} />

            <div>
              <strong>Medical Disclaimer</strong>
              <p>
                This application is for educational and research purposes.
                Its predictions are not a medical diagnosis or a substitute
                for professional medical advice. Consult a qualified
                healthcare professional for diagnosis and treatment.
              </p>
            </div>
          </footer>

          <div className="page-footer">
            <span>BrainCare AI · Brain MRI Classification System</span>
            <span>Powered by React, FastAPI and TensorFlow</span>
          </div>

        </div>
      </main>
    </div>
  );
}

export default App;
