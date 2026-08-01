# ML API with FastAPI and Docker

A simple machine learning API that predicts iris flower species using a trained Logistic Regression model, served with FastAPI and containerized with Docker.

## What This Project Does

- Trains a Logistic Regression model on the classic Iris dataset
- Serves predictions through a REST API built with FastAPI
- Runs inside a Docker container for easy deployment anywhere

## Tech Stack

- Python
- Scikit-learn
- FastAPI
- Docker

## Project Structure
ml-api-project/
├── train_model.py      # Trains and saves the ML model
├── main.py              # FastAPI application with prediction endpoint
├── model.pkl             # Saved trained model
├── requirement.txt       # Python dependencies
├── Dockerfile            # Docker build instructions
└── README.md

## How to Run

### Option 1: Run Locally

1. Install dependencies:
   pip install -r requirement.txt
2. Train the model:
   python train_model.py
3. Start the API:
   uvicorn main:app --reload
4. Open your browser at `http://127.0.0.1:8000/docs` to test the API.

### Option 2: Run with Docker

1. Build the Docker image:
   docker build -t ml-api .
2. Run the container:
   docker run -p 8000:8000 ml-api
3. Open your browser at `http://127.0.0.1:8000/docs` to test the API.

## API Endpoints

- `GET /` — Health check, confirms API is running
- `POST /predict` — Takes flower measurements and returns a predicted species

### Example Request

```json
{
"sepal_length": 5.1,
"sepal_width": 3.5,
"petal_length": 1.4,
"petal_width": 0.2
}
 Example Response
{
  "prediction": 0
}
