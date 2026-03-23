import type { PredictionResult } from "@/types/segmentation";

const DEFAULT_API_URL = "http://localhost:8000/segment";

export async function runInference(
  files: File[],
  apiUrl: string = DEFAULT_API_URL
): Promise<PredictionResult> {
  const formData = new FormData();
  files.forEach((file) => {
    formData.append("files", file);
  });

  const response = await fetch(apiUrl, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Inference failed (${response.status}): ${text}`);
  }

  return response.json();
}
