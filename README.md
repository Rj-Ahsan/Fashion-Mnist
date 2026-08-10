# Vercel Python Deployment

This repository is set up for deployment on Vercel using a Python serverless function.

## Included files

- `requirements.txt` — Python dependencies for Vercel.
- `vercel.json` — Vercel configuration for the Python function.
- `api/index.py` — Vercel API endpoint for model inference.
- `.gitignore` — ignores common Python artifacts.

## Deploy steps

1. Push this folder to GitHub.
2. Connect the repository to Vercel.
3. Vercel will detect the Python runtime using `requirements.txt`.
4. The deployed function is available at `/api/index`.

## Model setup

- Add a trained Keras model file named `model.h5` to the repository root.
- The endpoint expects Fashion MNIST grayscale images resized to `28x28`.
- If `model.h5` is missing, the deployed endpoint will return a load error.

## Example request

Send a POST request to `/api/index` with JSON:

```json
{
  "image": "<base64-encoded-image>",
  "top_k": 3
}
```

The response will contain the top predictions and probabilities.

## Important note

TensorFlow training is not well suited for Vercel serverless functions because of large package size and runtime limits. Use this deployment scaffold for inference only, and train the model locally before adding `model.h5` to the repository.
