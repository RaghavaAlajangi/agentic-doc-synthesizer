import React, { useState } from 'react';
import '../styles/CitationDisplay.css';

interface Citation {
  document_id: string;
  document_name: string;
  page_number?: string;
  section?: string;
  chunk_index: number;
  content_snippet: string;
  similarity_score: number;
  metadata: Record<string, any>;
}

interface CitationDisplayProps {
  citations: Citation[];
}

/**
 * CitationDisplay Component
 *
 * Displays source citations as beautiful reference cards with:
 * - Document name and location (page/section)
 * - Relevance score
 * - Content snippet preview
 */
export const CitationDisplay: React.FC<CitationDisplayProps> = ({
  citations,
}) => {
  const [expandedCitation, setExpandedCitation] = useState<number | null>(null);

  if (!citations || citations.length === 0) {
    return null;
  }

  const toggleExpanded = (index: number) => {
    setExpandedCitation(expandedCitation === index ? null : index);
  };

  return (
    <div className="citations-container">
      <div className="citations-header">
        <h3>📚 Source References ({citations.length})</h3>
        <p className="citations-subtitle">
          Based on retrieved documents and sections
        </p>
      </div>

      <div className="citations-list">
        {citations.map((citation, idx) => {
          const isExpanded = expandedCitation === idx;
          const relevancePercentage = Math.round(
            citation.similarity_score * 100
          );

          return (
            <div
              key={`${citation.document_id}-${idx}`}
              className={`citation-card ${isExpanded ? 'expanded' : ''}`}
            >
              {/* Citation Header - Always Visible */}
              <div
                className="citation-header"
                onClick={() => toggleExpanded(idx)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    toggleExpanded(idx);
                  }
                }}
              >
                <div className="citation-title-section">
                  <h4 className="citation-document-name">
                    📄 {citation.document_name}
                  </h4>
                  <div className="citation-location">
                    {citation.section && (
                      <span className="location-badge section-badge">
                        📍 {citation.section}
                      </span>
                    )}
                    {citation.page_number && (
                      <span className="location-badge page-badge">
                        📄 {citation.page_number}
                      </span>
                    )}
                    <span className="location-badge relevance-badge">
                      ✓ {relevancePercentage}% relevant
                    </span>
                  </div>
                </div>

                <div className="citation-toggle-icon">
                  {isExpanded ? '▼' : '▶'}
                </div>
              </div>

              {/* Citation Content - Expandable */}
              {isExpanded && (
                <div className="citation-content">
                  {/* Content Snippet */}
                  <div className="citation-snippet">
                    <h5>Content Preview</h5>
                    <p className="snippet-text">
                      {citation.content_snippet}
                      {citation.content_snippet.length >= 200 ? '...' : ''}
                    </p>
                  </div>

                  {/* Metadata Footer */}
                  <div className="citation-footer">
                    <span className="metadata-item">
                      Chunk {citation.chunk_index + 1}
                    </span>
                    <span className="metadata-item">
                      ID: {citation.document_id.substring(0, 8)}...
                    </span>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default CitationDisplay;
