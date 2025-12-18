import React, { useEffect, useState } from 'react';
import '../styles/UploadProgress.css';

interface UploadProgressProps {
  isUploading: boolean;
  progress: number;
  filename?: string;
}

export const UploadProgress: React.FC<UploadProgressProps> = ({
  isUploading,
  progress,
  filename,
}) => {
  const [steps, setSteps] = useState<
    {
      name: string;
      status: 'pending' | 'in-progress' | 'completed';
      icon: string;
    }[]
  >([
    { name: 'Reading file', status: 'pending', icon: '📄' },
    { name: 'Extracting pages', status: 'pending', icon: '📖' },
    { name: 'Creating chunks', status: 'pending', icon: '✂️' },
    { name: 'Summarizing sections', status: 'pending', icon: '📝' },
    { name: 'Extracting key info', status: 'pending', icon: '💡' },
    { name: 'Pushing to vector DB', status: 'pending', icon: '📤' },
    { name: 'Finalizing', status: 'pending', icon: '✅' },
  ]);

  useEffect(() => {
    if (!isUploading) {
      // Mark all as completed when done
      setSteps((prevSteps) =>
        prevSteps.map((step) => ({
          ...step,
          status: 'completed' as const,
        }))
      );
      return;
    }

    // Update steps based on progress
    const newSteps = [...steps];
    const stepsToComplete = Math.ceil((progress / 100) * steps.length);

    for (let i = 0; i < newSteps.length; i++) {
      if (i < stepsToComplete - 1) {
        newSteps[i].status = 'completed';
      } else if (i === stepsToComplete - 1 && stepsToComplete > 0) {
        newSteps[i].status = 'in-progress';
      } else {
        newSteps[i].status = 'pending';
      }
    }

    setSteps(newSteps);
  }, [progress, isUploading]);

  if (!isUploading && progress === 0) {
    return null;
  }

  return (
    <div className="upload-progress-container">
      <div className="progress-header">
        <h3>📤 Uploading Document</h3>
        {filename && <p className="filename">{filename}</p>}
      </div>

      <div className="progress-bar-container">
        <div className="progress-bar">
          <div
            className="progress-fill"
            style={{ width: `${progress}%` }}
          ></div>
        </div>
        <div className="progress-text">{progress}%</div>
      </div>

      <div className="steps-list">
        {steps.map((step, idx) => (
          <div
            key={idx}
            className={`step ${step.status}`}
          >
            <div className="step-icon">
              <span className="icon">{step.icon}</span>
              {step.status === 'completed' && (
                <span className="check">✓</span>
              )}
              {step.status === 'in-progress' && (
                <span className="spinner">⏳</span>
              )}
            </div>
            <div className="step-info">
              <p className="step-name">{step.name}</p>
              <p className="step-status">
                {step.status === 'completed'
                  ? 'Completed'
                  : step.status === 'in-progress'
                    ? 'Processing...'
                    : 'Pending'}
              </p>
            </div>
          </div>
        ))}
      </div>

      {progress === 100 && !isUploading && (
        <div className="completion-message">
          <p>✅ Upload completed successfully!</p>
        </div>
      )}
    </div>
  );
};

export default UploadProgress;
