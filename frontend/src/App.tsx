
import { useState } from "react";
import "./App.css";

type Module = "disease" | "nutrient";

type PredictionResult = {
  predicted_class?: string;
  confidence?: number;
  confidence_status?: string;
  class_probabilities?: Record<string, number>;
  disclaimer?: string;
  status?: string;
  quality?: {
    warnings?: string[];
    width?: number;
    height?: number;
  };
};

function App() {
  const [activeModule, setActiveModule] = useState<Module | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const goHome = () => {
    setActiveModule(null);
    setSelectedFile(null);
    setResult(null);
    setError("");
    setLoading(false);
  };

  const openModule = (module: Module) => {
    setActiveModule(module);
    setSelectedFile(null);
    setResult(null);
    setError("");
  };

  const handlePrediction = async () => {
    if (!selectedFile || !activeModule || loading) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const endpoint =
        activeModule === "disease"
          ? "/api/v1/disease/predict"
          : "/api/v1/nutrients/predict";

      const response = await fetch(endpoint, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : "Prediction failed. Please try another image."
        );
      }

      setResult(data);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to connect to the prediction server."
      );
    } finally {
      setLoading(false);
    }
  };

  const formatConfidence = (value?: number) =>
    typeof value === "number"
      ? `${(value * 100).toFixed(2)}%`
      : "Unavailable";

  return (
    <div className="app">
      <header className="navbar">
        <div className="brand" onClick={goHome}>
          🌱 PlantHealthAI
        </div>

        <nav>
          <button onClick={goHome}>Home</button>
          <a href="#modules">Detection Modules</a>
          <a href="#about">About</a>
        </nav>
      </header>

      <main>
        {!activeModule ? (
          <>
            <section className="hero">
              <span className="eyebrow">AI-POWERED AGRICULTURE</span>
              <h1>Smart Maize Plant Health Detection</h1>
              <p>
                An intelligent system for identifying maize leaf
                diseases and nutrient deficiencies using deep learning.
              </p>
              <a className="primary-btn" href="#modules">
                Explore Detection Modules →
              </a>
            </section>

            <section className="modules" id="modules">
              <div className="section-heading">
                <span className="eyebrow">OUR TECHNOLOGY</span>
                <h2>Plant Health Detection</h2>
                <p>Choose a module to analyse your maize leaf image.</p>
              </div>

              <div className="module-grid">
                <article className="module-card">
                  <div className="module-icon">🔬</div>
                  <span className="module-tag">MODULE 01</span>
                  <h3>Maize Disease Detection</h3>
                  <p>
                    Identify maize leaf diseases using a trained
                    deep learning classification model.
                  </p>
                  <button
                    className="primary-btn"
                    onClick={() => openModule("disease")}
                  >
                    Open Disease Detection →
                  </button>
                </article>

                <article className="module-card">
                  <div className="module-icon">🌿</div>
                  <span className="module-tag">MODULE 02</span>
                  <h3>Nutrient Deficiency Detection</h3>
                  <p>
                    Identify visual symptoms associated with five
                    maize nutrient categories.
                  </p>
                  <button
                    className="primary-btn"
                    onClick={() => openModule("nutrient")}
                  >
                    Open Nutrient Detection →
                  </button>
                </article>
              </div>
            </section>

            <section className="about" id="about">
              <span className="eyebrow">ABOUT THE PROJECT</span>
              <h2>Making Plant Health Analysis Smarter</h2>
              <p>
                PlantHealthAI combines deep learning and explainable
                AI techniques to support maize leaf analysis.
                It provides separate disease and nutrient detection
                capabilities.
              </p>
            </section>
          </>
        ) : (
          <section className="upload-section">
            <button className="back-btn" onClick={goHome}>
              ← Back to Dashboard
            </button>

            <span className="eyebrow">
              {activeModule === "disease"
                ? "DISEASE ANALYSIS"
                : "NUTRIENT ANALYSIS"}
            </span>

            <h1>
              {activeModule === "disease"
                ? "Maize Disease Detection"
                : "Maize Nutrient Deficiency Detection"}
            </h1>

            <p>Upload a clear maize leaf image to begin the analysis.</p>

            <div className="upload-box">
              <div className="upload-icon">📷</div>
              <h3>Upload Maize Leaf Image</h3>
              <p>Choose an image from your device.</p>

              <input
                type="file"
                accept="image/jpeg,image/png,image/webp"
                onChange={(event) => {
                  setSelectedFile(event.target.files?.[0] ?? null);
                  setResult(null);
                  setError("");
                }}
              />

              {selectedFile && (
                <p className="file-name">
                  Selected: {selectedFile.name}
                </p>
              )}

              <p className="upload-note">
                Supported formats: JPG, PNG and WEBP.
              </p>

              <button
                className="primary-btn analyze-btn"
                onClick={handlePrediction}
                disabled={!selectedFile || loading}
              >
                {loading ? "Analysing Image..." : "Analyse Image →"}
              </button>
            </div>

            {loading && (
              <p className="notice">
                Please wait while the AI model analyses your image.
              </p>
            )}

            {error && (
              <div className="error-box">
                <strong>Prediction Error</strong>
                <p>{error}</p>
              </div>
            )}

            {result && (
              <section className="result-card">
                <span className="eyebrow">ANALYSIS COMPLETE</span>
                <h2>Prediction Result</h2>

                <div className="prediction-highlight">
                  <span>Predicted Category</span>
                  <h3>{result.predicted_class ?? "Unavailable"}</h3>
                </div>

                <div className="result-stat">
                  <span>Model Confidence</span>
                  <strong>{formatConfidence(result.confidence)}</strong>
                </div>

                {result.confidence_status === "needs_review" && (
                  <p className="warning-box">
                    Low-confidence prediction. Please review the result
                    and consider uploading a clearer image.
                  </p>
                )}

                {result.class_probabilities && (
                  <div className="probabilities">
                    <h3>Class Probabilities</h3>
                    {Object.entries(result.class_probabilities)
                      .sort((a, b) => b[1] - a[1])
                      .map(([name, probability]) => (
                        <div className="probability-row" key={name}>
                          <div className="probability-label">
                            <span>{name.replaceAll("_", " ")}</span>
                            <strong>
                              {formatConfidence(probability)}
                            </strong>
                          </div>
                          <div className="probability-track">
                            <div
                              className="probability-fill"
                              style={{
                                width: `${Math.max(
                                  0,
                                  Math.min(100, probability * 100)
                                )}%`,
                              }}
                            />
                          </div>
                        </div>
                      ))}
                  </div>
                )}

                {result.quality?.warnings &&
                  result.quality.warnings.length > 0 && (
                    <div className="warning-box">
                      <strong>Image Quality Warnings</strong>
                      {result.quality.warnings.map((warning, index) => (
                        <p key={index}>{warning}</p>
                      ))}
                    </div>
                  )}

                {activeModule === "nutrient" && (
                  <p className="notice">
                    {result.disclaimer ??
                      "Predictions reflect visual dataset labels and are not laboratory-confirmed nutrient diagnoses."}
                  </p>
                )}

                <button
                  className="back-btn"
                  onClick={() => {
                    setSelectedFile(null);
                    setResult(null);
                    setError("");
                  }}
                >
                  Analyse Another Image
                </button>
              </section>
            )}
          </section>
        )}
      </main>

      <footer>
        <strong>PlantHealthAI</strong>
        <p>AI-based Maize Disease and Nutrient Deficiency Detection</p>
        <p>Academic Major Project · 2026</p>
      </footer>
    </div>
  );
}

export default App;
