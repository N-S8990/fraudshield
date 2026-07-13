import { useState } from 'react';
import Dashboard from './pages/Dashboard';
import LivePredict from './pages/LivePredict';
import './App.css';

type Tab = 'dashboard' | 'predict';

function App() {
  const [tab, setTab] = useState<Tab>('dashboard');

  return (
    <div className="app">
      <header>
        <h1>FraudShield</h1>
        <p className="tagline">Credit Card Fraud Detection &bull; ML-powered</p>
        <nav>
          <button className={tab === 'dashboard' ? 'active' : ''}
            onClick={() => setTab('dashboard')}>Dashboard</button>
          <button className={tab === 'predict' ? 'active' : ''}
            onClick={() => setTab('predict')}>Live Predict</button>
        </nav>
      </header>
      <main>
        {tab === 'dashboard' && <Dashboard />}
        {tab === 'predict' && <LivePredict />}
      </main>
      <footer>
        Built by  &bull; Nirav Sayanja
      </footer>
    </div>
  );
}

export default App;
