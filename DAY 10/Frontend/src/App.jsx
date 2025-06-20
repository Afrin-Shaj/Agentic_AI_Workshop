import React, { useState } from 'react';
import FileUploader from './components/FileUploader';
import JobDescriptionInput from './components/JobDescriptionInput';
import ResultDisplay from './components/ResultDisplay';
import { rankResumes } from './api';

const App = () => {
  const [files, setFiles] = useState([]);
  const [jd, setJd] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleProcess = async () => {
    if (!jd || files.length === 0) {
      alert('Please upload resumes and enter a job description.');
      return;
    }

    setLoading(true);
    try {
      const res = await rankResumes(jd, files);
      setResults(res);
    } catch (err) {
      alert('Error while processing resumes.');
      console.error(err);
    }
    setLoading(false);
  };

  return (
    <div className="container">
      <h2 className="title">📄 Resume Ranking App</h2>

      <div className="card">
        <FileUploader setFiles={setFiles} />
      </div>

      <div className="card">
        <JobDescriptionInput jd={jd} setJd={setJd} />
      </div>

      <div style={{ textAlign: 'center' }}>
        <button onClick={handleProcess} disabled={loading}>
          {loading ? 'Processing...' : '🚀 Process Resumes'}
        </button>
      </div>

      {results && (
        <div className="card">
          <ResultDisplay results={results} />
        </div>
      )}
    </div>
  );
};

export default App;
