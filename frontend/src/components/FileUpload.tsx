import React, { useEffect, useRef, useState } from 'react';
import { documentService } from '../services/api';
import '../styles/FileUpload.css';
import { UploadProgress } from './UploadProgress';

interface FileUploadProps {
  onUploadSuccess?: (response: any) => void;
  onUploadError?: (error: any) => void;
}

export const FileUpload: React.FC<FileUploadProps> = ({
  onUploadSuccess,
  onUploadError,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const sourceType: 'external' | 'internal' = 'external';
  const [uploadedFiles, setUploadedFiles] = useState<any[]>(
    []
  );
  const [uploadError, setUploadError] = useState<string | null>(
    null
  );
  const [isLoadingFiles, setIsLoadingFiles] = useState(true);

  // Load documents from server on mount
  useEffect(() => {
    loadDocuments();
  }, []);

  const loadDocuments = async () => {
    try {
      setIsLoadingFiles(true);
      const response = await documentService.listDocuments(
        sourceType
      );
      if (response && response.documents) {
        setUploadedFiles(response.documents);
      }
    } catch (error) {
      console.error('Failed to load documents:', error);
    } finally {
      setIsLoadingFiles(false);
    }
  };

  const reloadDocuments = async () => {
    await loadDocuments();
  };

  const handleFileSelect = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    const file = files[0];
    setUploadError(null);

    // Validate file type
    if (
      !file.name.toLowerCase().endsWith('.pdf') &&
      !file.name.toLowerCase().endsWith('.txt')
    ) {
      const error = new Error(
        'Only PDF and TXT files are supported'
      );
      setUploadError('Only PDF and TXT files are supported');
      if (onUploadError) {
        onUploadError(error);
      }
      return;
    }

    // Validate file size
    if (file.size > 10 * 1024 * 1024) {
      const error = new Error('File size exceeds 10MB limit');
      setUploadError('File size exceeds 10MB limit');
      if (onUploadError) {
        onUploadError(error);
      }
      return;
    }

    setIsUploading(true);
    setUploadProgress(0);

    try {
      // Simulate progress
      const progressInterval = setInterval(() => {
        setUploadProgress((prev) =>
          Math.min(prev + 10, 90)
        );
      }, 100);

      const response = await documentService.uploadDocument(
        file,
        sourceType
      );

      clearInterval(progressInterval);
      setUploadProgress(100);

      // Reload documents from server
      await reloadDocuments();

      if (onUploadSuccess) {
        onUploadSuccess(response);
      }

      // Reset
      setTimeout(() => {
        setIsUploading(false);
        setUploadProgress(0);
        if (fileInputRef.current) {
          fileInputRef.current.value = '';
        }
      }, 500);
    } catch (error) {
      console.error('Upload error:', error);
      const errorMsg =
        error instanceof Error
          ? error.message
          : 'Upload failed. Please try again.';
      setUploadError(errorMsg);
      if (onUploadError) {
        onUploadError(error);
      }
      setIsUploading(false);
      setUploadProgress(0);
    }
  };

  const handleDelete = async (documentId: string) => {
    try {
      await documentService.deleteDocument(
        documentId,
        sourceType
      );
      setUploadedFiles((prev) =>
        prev.filter((f) => f.document_id !== documentId)
      );
    } catch (error) {
      console.error('Delete error:', error);
    }
  };

  return (
    <div className="file-upload-container">
      <div className="upload-section">
        <h2>Upload Research Reports</h2>

        {uploadError && (
          <div className="error-message">
            ✗ {uploadError}
          </div>
        )}

        <div className="upload-area">
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading}
            className="upload-button"
          >
            {isUploading
              ? 'Uploading...'
              : 'Select PDF or TXT File'}
          </button>
          <input
            ref={fileInputRef}
            type="file"
            onChange={handleFileSelect}
            accept=".pdf,.txt"
            style={{ display: 'none' }}
            disabled={isUploading}
          />
        </div>

        <UploadProgress
          isUploading={isUploading}
          progress={uploadProgress}
          filename={uploadedFiles.length > 0 ? undefined : undefined}
        />

        {isUploading && uploadProgress > 0 && (
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{ width: `${uploadProgress}%` }}
            ></div>
            <span className="progress-text">
              {uploadProgress}%
            </span>
          </div>
        )}
      </div>

      {uploadedFiles.length > 0 && (
        <div className="uploaded-files">
          <h3>📄 Uploaded Documents ({uploadedFiles.length})</h3>
          <div className="files-list">
            {uploadedFiles.map((file) => (
              <div
                key={file.document_id || file.id}
                className="file-item"
              >
                <div className="file-info">
                  <strong>
                    {file.filename ||
                      file.metadata?.filename ||
                      file.id}
                  </strong>
                  <div className="file-details">
                    {file.file_size ? (
                      <small>
                        📦 {(
                          file.file_size / 1024
                        ).toFixed(2)} KB
                      </small>
                    ) : (
                      <small>
                        📁 Source: {file.source || sourceType}
                      </small>
                    )}
                    {file.stored_at && (
                      <small>
                        🕐 {new Date(file.stored_at).toLocaleDateString()}
                      </small>
                    )}
                  </div>
                </div>
                <button
                  onClick={() =>
                    handleDelete(file.document_id || file.id)
                  }
                  className="delete-button"
                >
                  ❌ Delete
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {!isLoadingFiles && uploadedFiles.length === 0 && (
        <div className="no-documents">
          <p>
            No documents uploaded yet. Upload a PDF or TXT file to
            get started.
          </p>
        </div>
      )}

      {isLoadingFiles && (
        <div className="loading">
          <p>Loading documents...</p>
        </div>
      )}
    </div>
  );
};

export default FileUpload;
