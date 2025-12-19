import React, { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { chatService, StreamEvent } from '../services/api';
import '../styles/Chat.css';
import { CitationDisplay } from './CitationDisplay';

interface Message {
  id: string;
  role: 'user' | 'assistant' | 'agent';
  content: string;
  timestamp: Date;
  agentThoughts?: any[];
  searchResults?: any[];
  citations?: any[];
  recommendations?: any[];
  isStreaming?: boolean;
  currentStep?: string;
  stepStatus?: 'pending' | 'in-progress' | 'completed';
}

export const Chat: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: inputValue,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // Create a placeholder for the streaming response
      const assistantId = (Date.now() + 1).toString();
      const assistantMessage: Message = {
        id: assistantId,
        role: 'assistant',
        content: '',
        timestamp: new Date(),
        isStreaming: true,
        agentThoughts: [],
        searchResults: [],
        citations: [],
        recommendations: [],
      };

      setMessages((prev) => [...prev, assistantMessage]);

      // Stream the response
      await chatService.streamChat(
        {
          query: inputValue,
        },
        (event: StreamEvent) => {
          setMessages((prev) => {
            const updated = [...prev];
            const msgIndex = updated.findIndex(
              (m) => m.id === assistantId
            );

            if (msgIndex !== -1) {
              const msg = updated[msgIndex];

              switch (event.event_type) {
                case 'agent_thought':
                  msg.agentThoughts = msg.agentThoughts || [];
                  msg.agentThoughts.push(event.data);
                  // Update current step for real-time view with tool info
                  const toolInfo = event.data.tool_used 
                    ? ` (using ${event.data.tool_used})`
                    : '';
                  msg.currentStep = (
                    `🤖 ${event.data.agent_name || event.data.agent} is thinking${toolInfo}`
                  );
                  msg.stepStatus = 'in-progress';
                  break;

                case 'search_result':
                  msg.searchResults = msg.searchResults || [];
                  msg.searchResults.push(event.data);
                  msg.currentStep = (
                    `Found ${msg.searchResults.length} document(s)`
                  );
                  msg.stepStatus = 'completed';
                  break;

                case 'recommendation':
                  msg.recommendations = msg.recommendations || [];
                  msg.recommendations.push(event.data);
                  msg.currentStep = (
                    `Extracted recommendation: `
                    + `${event.data.recommendation}`
                  );
                  msg.stepStatus = 'completed';
                  break;

                case 'final_response':
                  msg.content = event.data.response;
                  // ✅ Citations included in final_response at the end
                  msg.citations = event.data.citations || [];
                  msg.isStreaming = false;
                  msg.currentStep = 'Response generated';
                  msg.stepStatus = 'completed';
                  break;

                case 'error':
                  msg.content = `Error: ${event.data.error}`;
                  msg.isStreaming = false;
                  msg.currentStep = 'Error occurred';
                  msg.stepStatus = 'pending';
                  break;
              }
            }

            return updated;
          });
        },
        (error) => {
          console.error('Stream error:', error);
          setMessages((prev) => {
            const updated = [...prev];
            const msgIndex = updated.findIndex(
              (m) => m.id === assistantId
            );
            if (msgIndex !== -1) {
              updated[msgIndex].content = (
                'Error: Failed to get response'
              );
              updated[msgIndex].isStreaming = false;
            }
            return updated;
          });
        }
      );
    } catch (error) {
      console.error('Error sending message:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h1>Research Report Analyzer</h1>
      </div>

      <div className="messages-container">
        {messages.length === 0 ? (
          <div className="empty-state">
            <p>
              Start by uploading research reports or ask a question
              about cross-asset recommendations
            </p>
          </div>
        ) : (
          messages.map((message) => (
            <div
              key={message.id}
              className={`message message-${message.role}`}
            >
              <div className="message-header">
                <span className="message-role">
                  {message.role.toUpperCase()}
                </span>
                <span className="message-time">
                  {message.timestamp.toLocaleTimeString()}
                </span>
              </div>

              {message.agentThoughts && 
                message.agentThoughts.length > 0 && (
                <div className="agent-execution-flow">
                  <div className="execution-title">
                    🔄 Agent Execution Flow:
                  </div>
                  {message.agentThoughts.map(
                    (thought, idx) => {
                      const agentName = (
                        thought.agent || 
                        thought.agent_name || ''
                      );
                      const agentLower = (
                        agentName.toLowerCase()
                      );
                      const agentColor = {
                        'router': '#fef3c7',
                        'retrieval': '#dcfce7',
                        'extraction': '#f3e8ff',
                        'synthesis': '#fecdd3',
                      }[agentLower] || '#e0e7ff';
                      
                      return (
                        <div
                          key={idx}
                          className="agent-step"
                        >
                          <div 
                            className="step-badge"
                            style={{
                              backgroundColor: agentColor
                            }}
                          >
                            <span className="step-number">
                              {idx + 1}
                            </span>
                            <span className="agent-badge-name">
                              {agentName}
                            </span>
                            {thought.tool_used && (
                              <span className="agent-tool">
                                | {thought.tool_used}
                              </span>
                            )}
                          </div>
                          <div 
                            className="step-thought"
                          >
                            {thought.thought || 
                              thought.thought_text || 
                              'Processing...'}
                          </div>
                        </div>
                      );
                    }
                  )}
                </div>
              )}

              {message.content && (
                <div className="response-wrapper">
                  <div className="response-divider">
                    📝 Analysis Result
                  </div>
                  <div 
                    className="message-content markdown-content"
                  >
                    <ReactMarkdown
                    components={{
                      h1: ({ node, ...props }) => (
                        <h2 {...props} />
                      ),
                      h2: ({ node, ...props }) => (
                        <h3 {...props} />
                      ),
                      h3: ({ node, ...props }) => (
                        <h4 {...props} />
                      ),
                      a: ({ node, ...props }) => (
                        <a {...props} target="_blank" 
                           rel="noopener noreferrer" />
                      ),
                      code: ({ node, inline, ...props }) => (
                        <code
                          {...props}
                          style={{
                            background: inline
                              ? '#f0f0f0'
                              : '#1e1e1e',
                            color: inline
                              ? '#333'
                              : '#d4d4d4',
                            padding: inline
                              ? '2px 4px'
                              : '10px',
                            borderRadius: '4px',
                            display: inline
                              ? 'inline'
                              : 'block',
                          }}
                        />
                      ),
                      table: ({ node, ...props }) => (
                        <table
                          {...props}
                          style={{
                            borderCollapse: 'collapse',
                            width: '100%',
                            margin: '10px 0',
                          }}
                        />
                      ),
                      th: ({ node, ...props }) => (
                        <th
                          {...props}
                          style={{
                            border:
                              '1px solid #e0e0e0',
                            padding: '8px',
                            background:
                              '#f5f5f5',
                            textAlign: 'left',
                          }}
                        />
                      ),
                      td: ({ node, ...props }) => (
                        <td
                          {...props}
                          style={{
                            border:
                              '1px solid #e0e0e0',
                            padding: '8px',
                          }}
                        />
                      ),
                    }}
                  >
                    {message.content}
                  </ReactMarkdown>
                  </div>
                </div>
              )}

              {message.searchResults &&
                message.searchResults.length > 0 && (
                  <details
                    className="search-results"
                    open={
                      message.searchResults.length > 0
                    }
                  >
                    <summary className="results-summary">
                      📚 Retrieved Documents (
                      {(() => {
                        const uniqueDocs = new Set(
                          message.searchResults.map(
                            (r: any) => r.source
                          )
                        );
                        return uniqueDocs.size;
                      })()})
                    </summary>
                    <div className="results-list">
                      {(() => {
                        // Group results by source document
                        const grouped: Record<
                          string,
                          any[]
                        > = {};
                        message.searchResults.forEach(
                          (result: any) => {
                            if (!grouped[result.source]) {
                              grouped[result.source] = [];
                            }
                            grouped[result.source].push(
                              result
                            );
                          }
                        );

                        return Object.entries(grouped).map(
                          ([docName, chunks]) => (
                            <details
                              key={docName}
                              className="document-group"
                            >
                              <summary className="doc-summary">
                                📄 {docName} (
                                {chunks.length}{' '}
                                chunk{
                                  chunks.length > 1
                                    ? 's'
                                    : ''
                                }
                                )
                              </summary>
                              <div className="chunks-list">
                                {chunks.map(
                                  (chunk, idx) => {
                                    // Extract metadata from chunk
                                    const metadata = chunk.metadata || {};
                                    const chunk_metadata = chunk.chunk_metadata || metadata;
                                    
                                    // Basic metadata
                                    const page = metadata.page || chunk.page_number || '';
                                    const section = metadata.section || chunk.section || '';
                                    
                                    // Enhanced metadata
                                    const company = chunk_metadata.company_name || metadata.company_name || '';
                                    const reportType = chunk_metadata.report_type || metadata.report_type || '';
                                    const reportDate = chunk_metadata.report_date || metadata.report_date || '';
                                    const analyst = chunk_metadata.author_analyst || metadata.author_analyst || '';
                                    const rating = chunk_metadata.rating || metadata.rating || '';
                                    const targetPrice = chunk_metadata.target_price || metadata.target_price || '';
                                    
                                    return (
                                      <div
                                        key={idx}
                                        className="chunk-item"
                                      >
                                        {/* Source Attribution Card */}
                                        <div className="source-attribution">
                                          {/* Source Title and Relevance */}
                                          <div className="source-header">
                                            <div className="source-info">
                                              <h4 className="source-title">
                                                {metadata.filename ||
                                                  chunk_metadata.document_name ||
                                                  metadata.source ||
                                                  'Unknown Document'}
                                              </h4>
                                              <p className="source-relevance">
                                                Relevance: {(
                                                  chunk.similarity_score ?
                                                    chunk.similarity_score * 100 :
                                                    chunk.similarity * 100
                                                ).toFixed(1)}%
                                              </p>
                                            </div>
                                          </div>

                                          {/* Metadata Row */}
                                          <div className="metadata-row">
                                            {company && (
                                              <span className="metadata-item">
                                                <span className="metadata-label">Company:</span> {company}
                                              </span>
                                            )}
                                            {reportDate && (
                                              <span className="metadata-item">
                                                <span className="metadata-label">Year:</span> {reportDate}
                                              </span>
                                            )}
                                            {page && (
                                              <span className="metadata-item">
                                                <span className="metadata-label">Page:</span> {page}
                                              </span>
                                            )}
                                            {chunk_metadata.author_analyst && (
                                              <span className="metadata-item">
                                                <span className="metadata-label">Publisher:</span> {chunk_metadata.author_analyst}
                                              </span>
                                            )}
                                          </div>

                                          {/* Short Description */}
                                          {(reportType || rating || targetPrice) && (
                                            <div className="source-description">
                                              <div className="description-tags">
                                                {reportType && (
                                                  <span className="tag-small">{reportType}</span>
                                                )}
                                                {rating && (
                                                  <span className={`tag-small rating-${rating.toLowerCase()}`}>
                                                    {rating}
                                                  </span>
                                                )}
                                                {targetPrice && (
                                                  <span className="tag-small">{targetPrice}</span>
                                                )}
                                              </div>
                                            </div>
                                          )}
                                        </div>
                                      </div>
                                    );
                                  }
                                )}
                              </div>
                            </details>
                          )
                        );
                      })()}
                    </div>
                  </details>
                )}

              {message.recommendations &&
                message.recommendations.length > 0 && (
                  <details
                    className="recommendations"
                    open={
                      message.recommendations.length > 0
                    }
                  >
                    <summary className="rec-summary">
                      💡 Recommendations (
                      {message.recommendations.length})
                    </summary>
                    <div className="recommendations-list">
                      {message.recommendations.map(
                        (rec, idx) => (
                          <div
                            key={idx}
                            className="recommendation-item"
                          >
                            <div className="rec-header">
                              <strong>
                                {rec.asset_class}
                              </strong>
                              <span className="rec-status">
                                {rec.recommendation}
                              </span>
                            </div>
                            <p className="rec-source">
                              📌 {rec.source}
                            </p>
                            <small className="rec-confidence">
                              Confidence:{' '}
                              {(
                                rec.confidence * 100
                              ).toFixed(0)}
                              %
                            </small>
                          </div>
                        )
                      )}
                    </div>
                  </details>
                )}

              {message.citations &&
                message.citations.length > 0 && (
                  <CitationDisplay citations={message.citations} />
                )}

              {message.isStreaming && (
                <div className="streaming-indicator">
                  <span className="dot"></span>
                  <span className="dot"></span>
                  <span className="dot"></span>
                </div>
              )}
            </div>
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="input-container">
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyPress={(e) => {
            if (
              e.key === 'Enter' &&
              !e.shiftKey
            ) {
              e.preventDefault();
              handleSendMessage();
            }
          }}
          placeholder="Ask about research recommendations..."
          disabled={isLoading}
        />
        <button
          onClick={handleSendMessage}
          disabled={isLoading || !inputValue.trim()}
        >
          {isLoading ? 'Analyzing...' : 'Send'}
        </button>
      </div>
    </div>
  );
};

export default Chat;

