import axios from 'axios';

const api = axios.create({ baseURL: '/api/v1' });

export interface ModelMetrics {
  model: string;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number;
}

export interface PredictionResult {
  fraud_probability: number;
  prediction: number;
  model: string;
}

export type TransactionFeatures = Record<string, number>;

/** Fetch model comparison metrics */
export async function fetchModels(): Promise<ModelMetrics[]> {
  const { data } = await api.get('/models');
  return data.models ?? [];
}

/** Predict a single transaction */
export async function predictTransaction(features: TransactionFeatures): Promise<PredictionResult> {
  const { data } = await api.post('/predict', features);
  return data;
}
