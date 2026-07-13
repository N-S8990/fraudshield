import { useEffect, useState } from 'react';
import { fetchModels } from '../services/api';
import type { ModelMetrics } from '../services/api';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, Legend, CartesianGrid, ResponsiveContainer,
} from 'recharts';

export default function Dashboard() {
  const [models, setModels] = useState<ModelMetrics[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchModels()
      .then(setModels)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="loading">Loading model metrics…</div>;

  const chartData = models.map((m) => ({
    name: m.model.replace(/_/g, ' '),
    Precision: +(m.precision * 100).toFixed(1),
    Recall: +(m.recall * 100).toFixed(1),
    'F1 Score': +(m.f1 * 100).toFixed(1),
    'ROC-AUC': +(m.roc_auc * 100).toFixed(1),
  }));

  return (
    <div className="dashboard">
      <h2>Model Performance Comparison</h2>
      <p className="subtitle">
        Trained on Kaggle Credit Card Fraud dataset (284,807 transactions, 0.17% fraud rate)
        with SMOTE oversampling.
      </p>

      <ResponsiveContainer width="100%" height={400}>
        <BarChart data={chartData} barGap={4}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis domain={[0, 100]} tickFormatter={(v: number) => `${v}%`} />
          <Tooltip />
          <Legend />
          <Bar dataKey="Precision" fill="#8884d8" />
          <Bar dataKey="Recall" fill="#82ca9d" />
          <Bar dataKey="F1 Score" fill="#ffc658" />
          <Bar dataKey="ROC-AUC" fill="#ff7300" />
        </BarChart>
      </ResponsiveContainer>

      <div className="model-cards">
        {models.map((m) => (
          <div key={m.model} className="model-card">
            <h3>{m.model.replace(/_/g, ' ')}</h3>
            <table>
              <tbody>
                {(['precision', 'recall', 'f1', 'roc_auc'] as const).map((k) => (
                  <tr key={k}>
                    <td>{k.toUpperCase()}</td>
                    <td>{(m[k] * 100).toFixed(2)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ))}
      </div>

      <div className="insight-box">
        <strong>Key takeaway:</strong> Logistic Regression catches most frauds (91.8% recall)
        but flags many false positives. Random Forest is the most balanced (83.5% precision,
        82.7% recall). XGBoost offers the highest ROC-AUC (97.5%), making it the best
        default model for this pipeline.
      </div>
    </div>
  );
}
