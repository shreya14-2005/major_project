# PlantHealthAI Frontend Integration Plan

**Status:** Inspection and planning only. No application, model, dataset, report,
or frontend files were changed. Implementation requires approval and locating
the existing frontend project.

## 1. Existing frontend inspection

### Finding: frontend not present in the inspected workspace

The workspace root contains `datasets/`, `major_project/`, and a report text
file. The project directory contains ML training, evaluation, prediction,
testing, model, and explainability code. No frontend source files or frontend
project markers were found outside generated/virtual-environment directories:

- No React, Vue, Svelte, Angular, or other frontend source was identified.
- No `package.json`, frontend lockfile, `index.html`, or frontend build config
  was found.
- No app entry point, pages/components, router, navigation, upload widget,
  styling system, or existing frontend API client was found.
- No existing HTTP API/backend server was found.

**Framework and version: NOT VERIFIED (no frontend present).** The inspected
workspace therefore does not establish a frontend design system or existing
routes to extend. Do not create a replacement frontend or choose new UI styling
without locating and inspecting the intended frontend repository/folder.

## 2. Existing backend integration points

### Disease module

- Active model: `models/maize_disease_model.keras`; verified architecture is
  MobileNetV2, input 224×224 RGB, four output classes.
- Mapping: `models/maize_disease_classes.json`.
- Active inference helper: `testing/disease_inference.py`. It exposes
  `load_model_bundle()`, `read_rgb_input()`, and `predict_image()`. The helper
  loads only the disease model and mapping. `predict_image()` returns the
  predicted class, confidence, all class probabilities, review status, quality
  warnings, OOD status, and optional explanation heatmaps.
- Training-matched preprocessing: EXIF orientation correction, RGB, 224×224
  bilinear resize, float32 raw pixels in [0,255]; the saved model applies
  MobileNetV2 rescaling (1/127.5, offset -1).
- Existing explanation support: `explainability/gradcam.py` and
  `explainability/gradcam_plus_plus.py`. `build_explanation_context()` and
  `generate_explanations()` in the disease helper calculate the predicted
  class's Grad-CAM and Grad-CAM++ maps. `make_overlay()` produces a resized
  overlay. Heatmaps are approximate and must not be presented as proof of
  biological diagnosis.
- There is no separate disease prediction CLI under `predictions/`. Avoid
  invoking `testing/test_disease_module.py` as an API: it is a test/report
  runner and can write evaluation artifacts.
- Previously reported disease test accuracy is 95.22% (598/628). The
  EfficientNetB0/96.18% historical claim remains NOT VERIFIED; do not label
  the active model or expose that claim as verified.

### Nutrient module

- Active model: `models/nutrient_detection/nutrient_model.keras`; verified
  architecture is EfficientNetB0, input 224×224 RGB, five output classes.
- Mapping: `models/nutrient_detection/nutrient_classes.json`.
- Standalone inference: `predictions/nutrient_detection/predict_nutrient.py`,
  callable as `predict_nutrient_deficiency(image_input)`. It returns predicted
  class, confidence, all five class probabilities, and a biological-diagnosis
  disclaimer. It loads only the nutrient model/mapping.
- Preprocessing: RGB, Pillow LANCZOS resize to 224×224, float32 raw pixels in
  [0,255]; EfficientNet normalization is embedded in the saved model. Do not
  share the disease resize or normalization pipeline.
- Grad-CAM: `explainability/nutrient_gradcam/gradcam_nutrient.py`. It exposes
  `load_model_and_classes()`, `build_gradcam_model()`, `read_model_input()`,
  and `save_explanation()`. The final convolutional layer is `top_conv`
  (7×7×1280); target class defaults to the prediction and can be selected.
  `save_explanation()` writes PNG and JSON files to a results directory and
  can disable interactive display.
- Existing reported held-out accuracy is 92.96% (66/71); do not rerun
  evaluation as part of endpoint implementation.
- Nutrient labels are visually assigned field-survey labels and are explicitly
  documented as biologically unverified. Preserve this distinction in UI copy.

### Runtime and server availability

- A project-local `.venv` now exists at `major_project/.venv`, using Python
  3.12.0 with system site packages. Verified runtime: TensorFlow 2.18.1,
  Keras 3.15.1, OpenCV 5.0.0, pytest 9.2.0.
- No FastAPI, Flask, Starlette, or Uvicorn package was found in the active
  environment. No backend server or API routes currently exist.
- Disease inference and explainability depend on TensorFlow, NumPy, Pillow,
  Matplotlib, and OpenCV. Nutrient prediction/Grad-CAM depend on TensorFlow,
  NumPy, Pillow, and Matplotlib.
- The project has no dependency manifest. The existing `.venv` can be used
  for a future local API, but dependencies should be documented in an
  approved, project-local backend manifest rather than installed globally.

## 3. Recommended integration architecture

Keep the two classifiers as separate services behind one HTTP server process:

```text
Existing frontend (location/framework to be supplied)
  ├─ Disease page  ── POST /api/v1/disease/predict
  └─ Nutrient page  ── POST /api/v1/nutrients/predict
                              │
                    API boundary / upload validation
                       ├─ Disease service
                       │    ├─ disease model + mapping
                       │    ├─ disease preprocessing
                       │    ├─ disease prediction
                       │    ├─ Grad-CAM
                       │    └─ Grad-CAM++
                       └─ Nutrient service
                            ├─ nutrient model + mapping
                            ├─ nutrient preprocessing
                            ├─ nutrient prediction
                            └─ nutrient Grad-CAM
```

Suggested implementation is FastAPI with Uvicorn, subject to approval. It is
not installed yet. Use a single application process with each model loaded
once at startup and held in its own service/bundle. Each service must use
explicit model and mapping paths and its existing preprocessing functions.
Avoid shared model globals, a generic preprocessing function, or fallback
from one classifier to the other.

### Proposed endpoints

#### `POST /api/v1/disease/predict`

- Multipart upload field: `image`.
- Optional `include_explanations` boolean, default false or true by product
  decision; when enabled, generate both Grad-CAM and Grad-CAM++.
- Response: `predicted_class`, `confidence`, `class_probabilities`,
  `confidence_status`, `out_of_distribution_status`, `quality`, and
  `explanations`.
- For each explanation, return a viewable overlay (for example a data URL or
  a short-lived image URL) and optional heatmap. Convert NumPy arrays to JSON-
  serializable values or image bytes; do not return raw NumPy arrays.
- Preserve `out_of_distribution_status: not_assessed`; confidence is not an
  OOD detector.

#### `POST /api/v1/nutrients/predict`

- Multipart upload field: `image`.
- Optional `include_explanation` boolean and `target_class` string. If omitted,
  explain the predicted class; validate explicit targets against the five
  nutrient classes.
- Response: `predicted_class`, `confidence`, `class_probabilities`,
  `target_class`, optional Grad-CAM overlay/image, and the biological-label
  disclaimer.
- Return explanation bytes or a short-lived URL instead of writing every API
  request into the persistent `explainability/nutrient_gradcam/results/`
  folder. Keep the existing CLI behavior and stored examples unchanged.

#### Optional `GET /api/v1/health`

Return process readiness and whether both model bundles loaded. Avoid
including filesystem details or model internals in public error responses.

### Upload, errors, and execution

- Validate content type, file extension, byte size, successful image decoding,
  and pixel dimensions before inference. Enforce an upload size limit and
  reject malformed or corrupted images with a clear 4xx response.
- Use a unique temporary file or decoded image object for each request;
  clean up temporary files in `finally`. Do not trust client-supplied paths.
- Map known validation errors to safe user-facing messages. Log detailed
  server-side errors without returning stack traces.
- Model loading failures should fail application readiness/startup explicitly,
  not silently fall back to another module/model.
- TensorFlow inference and Grad-CAM are CPU/GPU-bound. Run them off the
  asynchronous event loop (threadpool/background worker) and limit concurrent
  explanation requests to avoid CPU/RAM exhaustion.
- Configure CORS only for the actual frontend origin when known. Same-origin
  deployment is simpler and avoids broad wildcard CORS.
- Do not save uploads or predictions permanently by default. If history or
  persistent images are later required, define retention and access controls.

## 4. Proposed frontend changes (after locating existing app)

Preserve the existing framework, navigation, component conventions, and
design tokens once inspected. Add routes/pages to the current app rather than
creating another frontend:

1. **Disease detection page:** image selector/drop zone, preview, submit
   action, loading/error states, predicted disease and confidence, per-class
   probabilities, quality/review indicators, and optional Grad-CAM / Grad-CAM++
   image tabs or side-by-side panels.
2. **Nutrient deficiency page:** same upload/preview interaction, but separate
   result view and its own five-class probabilities and Grad-CAM panel. Include
   a clear notice that labels/predictions are not biologically confirmed.
3. **Navigation:** add two links/routes to the existing navigation after
   confirming its current routing and active-link patterns.
4. **Shared upload presentation only:** a reusable visual upload/preview
   component may be appropriate, but keep API requests, status, result types,
   and image-processing behavior module-specific.
5. **Loading and errors:** disable duplicate submissions; show upload,
   prediction, and explanation progress; allow retry; handle unsupported,
   oversized, unreadable, network, server, and model-unavailable errors.
6. **Accessibility:** keyboard-accessible file selection, visible focus,
   descriptive image alt text, and announced loading/error/result states.
7. **Confidence wording:** describe confidence as model output, not diagnostic
   certainty. Do not present 95.22%/92.96% as per-image confidence or imply
   guaranteed accuracy.

Exact frontend file paths and component names are intentionally NOT VERIFIED;
no frontend was found in this workspace.

## 5. File-by-file implementation plan

### Existing files to reuse (do not change without approval)

| File | Integration role |
|---|---|
| `testing/disease_inference.py` | Disease model bundle, input validation/preprocessing, prediction, optional explanations, overlays. Its `testing` name reflects current structure; consider a dedicated inference package only if an approved refactor is needed. |
| `explainability/gradcam.py` | Disease Grad-CAM model and heatmap functions. |
| `explainability/gradcam_plus_plus.py` | Disease Grad-CAM++ function. |
| `predictions/nutrient_detection/predict_nutrient.py` | Nutrient model loading, preprocessing, probabilities, disclaimer. |
| `explainability/nutrient_gradcam/gradcam_nutrient.py` | Nutrient target-aware Grad-CAM and visualization utilities. |
| `models/maize_disease_model.keras` and `models/maize_disease_classes.json` | Disease-only weights and mapping. Active architecture is MobileNetV2. |
| `models/nutrient_detection/nutrient_model.keras` and `models/nutrient_detection/nutrient_classes.json` | Nutrient-only weights and mapping. Active architecture is EfficientNetB0. |

### Proposed new backend files (not created)

| Proposed file | Responsibility |
|---|---|
| `api/main.py` | FastAPI app, lifecycle/startup, health and two independent prediction routes, CORS configuration. |
| `api/schemas.py` | Request/response models for disease and nutrient results, with distinct types. |
| `api/services/disease.py` | Load disease-only bundle once; validate class mapping; invoke disease preprocessing, prediction, Grad-CAM and Grad-CAM++; serialize overlay images. |
| `api/services/nutrients.py` | Load nutrient-only bundle once; invoke nutrient preprocessing, prediction and optional target-aware Grad-CAM; serialize overlay images and disclaimer. |
| `api/image_uploads.py` | Upload byte limits, decode/validation, safe temporary-file lifecycle, MIME checks. |
| `api/requirements.txt` or the project's approved manifest | Pin/document compatible server dependencies and reuse the verified TensorFlow/Keras environment without unnecessary version changes. |
| `api/tests/test_disease_api.py` | Disease endpoint contract, invalid upload, result shape, and explanation checks using a small fixture. |
| `api/tests/test_nutrients_api.py` | Nutrient endpoint contract, five probabilities, explicit target, invalid upload, and disclaimer checks. |
| `api/tests/test_model_isolation.py` | Assert each service loads its own model/mapping and never invokes the other service. |

Actual frontend paths must be added to this plan after the existing frontend
location is supplied and inspected. No frontend project should be scaffolded.

## 6. Integration testing plan (for implementation phase)

1. **Model isolation:** assert each service's configured model path and class
   mapping; inspect startup behavior and verify one endpoint cannot call the
   other model.
2. **Preprocessing parity:** compare endpoint/service preprocessing against
   the established disease and nutrient helper functions on RGB, grayscale,
   EXIF-oriented, and differently sized fixtures.
3. **Prediction contracts:** verify four disease or five nutrient finite
   probabilities sum to approximately one; class names follow the model's
   mapping; confidence equals the predicted class probability.
4. **Explanation contracts:** verify disease Grad-CAM and Grad-CAM++ and
   nutrient predicted/explicit target Grad-CAM return finite, nonblank images
   at original image dimensions. Test invalid target classes.
5. **Upload failures:** missing file, empty file, unsupported format, corrupt
   image, oversized upload, and excessive dimensions produce safe 4xx results.
6. **Lifecycle/errors:** missing/unreadable model or class mapping prevents
   readiness and gives a logged server error; no silent model fallback occurs.
7. **Persistence/isolation:** requests leave no permanent uploaded files,
   do not overwrite existing Grad-CAM/evaluation outputs, and remove temporary
   artifacts after success or failure.
8. **Frontend integration:** mock API success, loading, validation errors,
   server unavailable, and explanation image rendering; verify navigation and
   responsive behavior within the existing design system.
9. **No held-out reevaluation:** API and integration tests use synthetic
   fixtures or explicitly selected non-evaluation images; do not run the
   aggregate held-out test evaluation.

## 7. Technical limitations and approval gates

- **Blocking discovery:** no frontend exists in the inspected workspace.
  Supply/open the existing frontend directory or repository before naming its
  framework, routes, files, components, or design system. The frontend is not
  assumed to be absent elsewhere on the user's machine.
- No API framework/server is currently installed; FastAPI is a proposal, not
  an existing dependency. Add it only after implementation approval, inside
  the project-local `.venv`.
- Disease and nutrient Grad-CAM maps originate at 7×7 resolution; resizing
  makes display overlays, not higher-detail explanations.
- Disease OOD detection is not assessed; quality warnings and confidence do
  not identify unrelated or non-maize images.
- Nutrient labels are unverified biological field-survey annotations. Keep a
  visible disclaimer and do not claim a confirmed agronomic diagnosis.
- Disease model history discrepancy remains: the active model is MobileNetV2
  and the supported test report is 95.22%; the historical EfficientNetB0 /
  96.18% claim is not verified. Integration must use the actual active model
  and must not display the unsupported claim.
- Existing nutrient Grad-CAM saves output to a persistent results directory
  by default. API usage should provide an isolated temporary output directory
  or add an approved in-memory image-return path in a later implementation.

**Stop point:** This document is the requested plan only. No endpoints,
frontend changes, dependency installations, or other code changes were made
for this task. Wait for approval and the location of the existing frontend
before implementation.
