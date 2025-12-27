import { useState } from 'react';
import './App.css';
import Chat from './components/Chat';
import FileUpload from './components/FileUpload';

function App() {
  const [showUpload, setShowUpload] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState<
    string | null
  >(null);

  const handleUploadSuccess = (response: any) => {
    setUploadSuccess(
      `Successfully uploaded: ${response.metadata.filename}`
    );
    setTimeout(() => setUploadSuccess(null), 5000);
  };

  const handleUploadError = (error: any) => {
    console.error('Upload failed:', error);
  };

  return (
    <div className="App">
      <div className="app-layout">
        <aside className="sidebar">
          <div className="sidebar-header">
            <h2>FinLens</h2>
            <p className="subtitle">
              Financial report analysis
            </p>
          </div>

          <button
            className="upload-toggle"
            onClick={() =>
              setShowUpload(!showUpload)
            }
          >
            {showUpload ? 'Hide' : 'Show'} Upload
          </button>

          {showUpload && (
            <FileUpload
              onUploadSuccess={handleUploadSuccess}
              onUploadError={handleUploadError}
            />
          )}

          {uploadSuccess && (
            <div className="success-message">
              ✓ {uploadSuccess}
            </div>
          )}

          <div className="sidebar-footer">
            <div className="info-section">
              <h4>About</h4>
              <p>
                Multi-agent AI assistant for analyzing
                financial reports.
              </p>
            </div>

            <div className="info-section">
              <h4>Features</h4>
              <ul>
                <li>Upload PDF/TXT research reports</li>
                <li>Ask questions about recommendations</li>
                <li>View agent reasoning</li>
                <li>Compare external vs internal views</li>
              </ul>
            </div>

            <div className="info-section">
              <h4>Supported Assets</h4>
              <ul>
                <li>Equities</li>
                <li>Fixed Income</li>
                <li>Multi-Asset</li>
                <li>FX & Commodities</li>
              </ul>
            </div>
          </div>
        </aside>

        <main className="main-content">
          <Chat />
        </main>
      </div>
    </div>
  );
}

export default App;

