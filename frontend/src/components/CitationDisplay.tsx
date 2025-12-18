import React, { useState } from 'react';
import '../styles/CitationDisplay.css';

interface VitalInfo {
  sectors?: string[];
  recommendations?: string[];
  metrics?: Record<string, string | number>;
  risks?: string[];
}

interface Citation {
  document_id: string;
  document_name: string;
  page_number?: string;
  section?: string;
  section_summary?: string;
  chunk_index: number;
  total_chunks?: number;
  content_snippet: string;
  vital_info?: VitalInfo;
  similarity_score: number;
}

interface CitationDisplayProps {
  citations: Citation[];
}

/**
 * CitationDisplay Component
 *
 * Displays source citations as beautiful reference cards with:
 * - Document name and location (page/section)
 * - Section summary for quick context
 * - Vital information extracted (sectors, recommendations, metrics, risks)
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

  const formatVitalInfo = (vital_info: VitalInfo | undefined) => {
    if (!vital_info) return null;

    const items = [];

    if (vital_info.sectors && vital_info.sectors.length > 0) {
      items.push({
        label: 'Sectors',
        value: vital_info.sectors.join(', '),
        icon: '🏢',
      });
    }

    if (vital_info.recommendations && vital_info.recommendations.length > 0) {
      items.push({
        label: 'Recommendations',
        value: vital_info.recommendations.join(', '),
        icon: '💡',
      });
    }

    if (vital_info.metrics && Object.keys(vital_info.metrics).length > 0) {
      const metricsStr = Object.entries(vital_info.metrics)
        .map(([key, val]) => `${key}: ${val}`)
        .join(', ');
      items.push({
        label: 'Metrics',
        value: metricsStr,
        icon: '📊',
      });
    }

    if (vital_info.risks && vital_info.risks.length > 0) {
      items.push({
        label: 'Risks',
        value: vital_info.risks.join(', '),
        icon: '⚠️',
      });
    }

    return items;
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
          const vitalItems = formatVitalInfo(citation.vital_info);
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
                  {/* Section Summary */}
                  {citation.section_summary && (
                    <div className="citation-summary">
                      <h5>Summary</h5>
                      <p>{citation.section_summary}</p>
                    </div>
                  )}

                  {/* Vital Information */}
                  {vitalItems && vitalItems.length > 0 && (
                    <div className="citation-vital-info">
                      <h5>Key Information</h5>
                      <div className="vital-items">
                        {vitalItems.map((item, itemIdx) => (
                          <div key={itemIdx} className="vital-item">
                            <span className="vital-icon">{item.icon}</span>
                            <div className="vital-content">
                              <span className="vital-label">{item.label}:</span>
                              <span className="vital-value">{item.value}</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

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
                      {citation.total_chunks
                        ? ` of ${citation.total_chunks}`
                        : ''}
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
