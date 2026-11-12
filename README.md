# PlantHealthAI
### AI-Powered Maize Disease and Nutrient Deficiency Detection

PlantHealthAI is a machine-learning-based application designed to assist
in identifying visible maize leaf diseases and nutrient deficiency symptoms
using uploaded leaf images.

The project contains two independent prediction modules:
1. Maize Disease Detection
2. Maize Nutrient Deficiency Detection

---

## Project Objectives

- Identify maize leaf disease categories from images.
- Identify visual nutrient deficiency categories.
- Provide prediction confidence and class probabilities.
- Explore model explainability using Grad-CAM.
- Develop a user-friendly web interface for image submission
  and prediction results.

## Main Features

### 1. Maize Disease Detection

- Model architecture: MobileNetV2
- Input image size: 224 x 224 RGB
- Number of output classes: 4
- Test accuracy: 95.22% (598/628)
- Explainability: Grad-CAM and Grad-CAM++

### 2. Maize Nutrient Deficiency Detection

- Model architecture: EfficientNetB0
- Input image size: 224 x 224 RGB
- Number of output classes: 5
- Test accuracy: 92.96% (66/71)
- Explainability: Nutrient Grad-CAM

Nutrient classes:
- Healthy
- Magnesium
- Nitrogen
- Phosphorus
- Potassium

## Model Evaluation

| Module | Architecture | Test Accuracy |
|---|---|---:|
| Disease Detection | MobileNetV2 | 95.22% |
| Nutrient Detection | EfficientNetB0 | 92.96% |

These results refer to the respective held-out test datasets.
They should not be interpreted as guaranteed real-world accuracy.

## Technology Stack

### Machine Learning and Backend
- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Pillow
- Scikit-learn

### Frontend
- React
- TypeScript
- Vite
- HTML
- CSS

### Explainability
- Grad-CAM
- Grad-CAM++

## Project Structure

```text
PlantHealthAI/
|
|-- frontend/
|   |-- src/
|   |-- public/
|   |-- package.json
|   |-- package-lock.json
|   |-- vite.config.ts
|   `-- README.md
|
|-- backend/
|   |-- datasets/
|   |-- models/
|   |-- training/
|   |-- predictions/
|   |-- explainability/
|   |-- evaluation/
|   |-- testing/
|   |-- outputs/
|   |-- .venv/
|   `-- .dist/
|
|-- docs/
|   `-- reports/
|
|-- .gitignore
`-- README.md
```

## Running the Frontend

Open a terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL displayed by Vite, usually:

http://localhost:5173/

## Backend Status

The machine-learning modules and their inference and explainability
components have been developed and tested.

The FastAPI integration layer is planned but has not yet been
implemented. The current React frontend provides the user interface
and image selection; prediction API integration is a subsequent step.

The existing Python virtual environment is located at:

`backend/.venv/`

## Dataset Information and Limitations

The nutrient deficiency dataset contains five visually labelled
categories. These labels are based on image-folder annotations and
are not laboratory-confirmed nutrient measurements.

Therefore, nutrient predictions should be treated as visual
classification results and not as definitive agronomic diagnoses.

Model performance may vary with lighting, image quality, leaf
orientation, camera conditions, and real-world field variation.

PlantHealthAI is an academic project and should not replace
professional agricultural or laboratory assessment.

## Project Documentation

Detailed development and verification reports are available in:

`docs/reports/`

## Future Development

- Integrate the React frontend with a FastAPI backend.
- Provide independent prediction endpoints for both modules.
- Display prediction confidence and class probabilities.
- Display Grad-CAM explanations.
- Improve input validation and error handling.
- Complete end-to-end application testing.

## Project Status

Machine-learning development: Completed

Frontend dashboard: Initial implementation completed

Backend API integration: Planned

End-to-end web application: In progress

---

**PlantHealthAI — Supporting smarter maize plant health assessment through AI.**
