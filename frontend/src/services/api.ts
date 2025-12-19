import axios, { AxiosInstance } from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Create axios instance
const api: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Interfaces for type safety
export interface ChatRequest {
  query: string;
  conversation_id?: string;
}

export interface AgentThought {
  agent_name: string;
  thought: string;
  timestamp: string;
  tool_used?: string;
  tool_input?: Record<string, any>;
  tool_output?: string;
}

export interface SearchResult {
  document_id: string;
  content: string;
  similarity_score: number;
  metadata: Record<string, any>;
}

export interface Citation {
  document_id: string;
  document_name: string;
  page_number?: string;
  section?: string;
  chunk_index: number;
  content_snippet: string;
  similarity_score: number;
  metadata: Record<string, any>;
}

export interface ChatResponse {
  response: string;
  agent_thoughts: AgentThought[];
  citations: Citation[];
  conversation_id?: string;
}

export interface StreamEvent {
  event_type: string;
  data: Record<string, any>;
  timestamp: string;
}

export interface DocumentUploadResponse {
  success: boolean;
  document_id: string;
  message: string;
  metadata: {
    filename: string;
    upload_timestamp: string;
    document_type: string;
    source_type: string;
    file_size: number;
  };
  chunks_stored: number;
}

// Chat service
export const chatService = {
  /**
   * Send a chat message and get a response
   */
  async sendMessage(request: ChatRequest): Promise<ChatResponse> {
    try {
      const response = await api.post<ChatResponse>('/chat', request);
      return response.data;
    } catch (error) {
      console.error('Error sending message:', error);
      throw error;
    }
  },

  /**
   * Stream a chat response with agent thoughts
   */
  async streamChat(
    request: ChatRequest,
    onEvent: (event: StreamEvent) => void,
    onError?: (error: Error) => void
  ): Promise<void> {
    try {
      const response = await fetch(
        `${API_BASE_URL}/chat/stream`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(request),
        }
      );

      if (!response.ok) {
        throw new Error(
          `HTTP error! status: ${response.status}`
        );
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('Response body is not readable');
      }

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');

        // Keep the last incomplete line in buffer
        buffer = lines[lines.length - 1];

        for (let i = 0; i < lines.length - 1; i++) {
          const line = lines[i];
          if (line.startsWith('data: ')) {
            try {
              const eventData = JSON.parse(
                line.slice(6)
              );
              const streamEvent: StreamEvent = {
                event_type: eventData.event_type,
                data: eventData.data,
                timestamp: new Date().toISOString(),
              };
              onEvent(streamEvent);
            } catch (e) {
              console.error(
                'Error parsing stream event:',
                e
              );
            }
          }
        }
      }
    } catch (error) {
      console.error('Error in stream chat:', error);
      if (onError) {
        onError(
          error instanceof Error
            ? error
            : new Error('Stream failed')
        );
      }
    }
  },
};

// Document service
export const documentService = {
  /**
   * Upload a PDF or text document
   */
  async uploadDocument(
    file: File,
    sourceType: string = 'external'
  ): Promise<DocumentUploadResponse> {
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('source_type', sourceType);

      const response = await api.post<DocumentUploadResponse>(
        '/upload-document',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );

      return response.data;
    } catch (error) {
      console.error('Error uploading document:', error);
      throw error;
    }
  },

  /**
   * Get list of all uploaded documents
   */
  async listDocuments(sourceType?: string): Promise<any> {
    try {
      const params = sourceType ? `?source_type=${sourceType}` : '';
      const response = await api.get(`/documents${params}`);
      return response.data;
    } catch (error) {
      console.error('Error listing documents:', error);
      throw error;
    }
  },

  /**
   * Delete a document by ID
   */
  async deleteDocument(
    documentId: string,
    sourceType: string = 'external'
  ): Promise<void> {
    try {
      await api.delete(`/documents/${documentId}`, {
        params: { source_type: sourceType },
      });
    } catch (error) {
      console.error('Error deleting document:', error);
      throw error;
    }
  },
};

// Health check
export const healthService = {
  /**
   * Check backend health
   */
  async checkHealth(): Promise<any> {
    try {
      const response = await api.get('/health');
      return response.data;
    } catch (error) {
      console.error('Health check failed:', error);
      throw error;
    }
  },
};

export default api;

