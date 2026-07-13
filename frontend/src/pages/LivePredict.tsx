import { useState } from 'react';
import { predictTransaction } from '../services/api';
import type { TransactionFeatures } from '../services/api';

const V_FEATURES = Array.from({ length: 28 }, (_, i) => `v${i + 1}`);

const defaultFeatures: TransactionFeatures = {
  time: 0, amount: 0,
  ...Object.fromEntries(V_FEATURES.map((v) => [v, 0])),
};

export default function LivePredict() {
  const [features, setFeatures] = useState(defaultFeatures);
  const [result, setResult] = useState<{ prob: number; pred: number } | null>(null);
  const [loading, setLoading] = useState(false);

  const update = (key: string, value: string) => {
    setFeatures((prev) => ({ ...prev, [key]: parseFloat(value) || 0 }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await predictTransaction(features);
      setResult({ prob: res.fraud_probability, pred: res.prediction });
    } catch (err) {
      console.error(err);
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="live-predict">
      <h2>Live Fraud Prediction</h2>
      <p className="subtitle">
        Enter transaction features below to check if it's fraudulent.
        V1–V28 are PCA components from the original dataset.
      </p>

      <form onSubmit={handleSubmit} className="predict-form">
        <fieldset>
          <legend>Core Features</legend>
          <label>
            Time (seconds)
            <input type="number" step="0.01" value={features.time}
              onChange={(e) => update('time', e.target.value)} />
          </label>
          <label>
            Amount ($)
            <input type="number" step="0.01" value={features.amount}
              onChange={(e) => update('amount', e.target.value)} />
          </label>
        </fieldset>

        <details>
          <summary>V1–V28 (PCA components) — optional, defaults to 0</summary>
          <div className="v-grid">
            {V_FEATURES.map((v) => (
              <label key={v}>
                {v}
                <input type="number" step="0.01" value={features[v]}
                  onChange={(e) => update(v, e.target.value)} />
              </label>
            ))}
          </div>
        </details>

        <button type="submit" disabled={loading}>
          {loading ? 'Predicting…' : 'Run Prediction'}
        </button>
      </form>

      {result && (
        <div className={`result-card ${result.pred === 1 ? 'fraud' : 'legit'}`}>
          <h3>{result.pred === 1 ? '⚠️ Fraud Detected' : '✅ Legitimate Transaction'}</h3>
          <p>
            Fraud probability: <strong>{(result.prob * 100).toFixed(2)}%</strong>
          </p>
          {result.pred === 1 && (
            <p className="note">This transaction has been flagged as potentially fraudulent.</p>
          )}
        </div>
      )}
    </div>
  );
}
