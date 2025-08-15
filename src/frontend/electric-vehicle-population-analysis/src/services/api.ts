import axios, { AxiosInstance, AxiosResponse } from 'axios';

// Types
export interface EVData {
  totalCount: number;
  bevCount: number;
  phevCount: number;
  dataQuality: {
    completeness: number;
    accuracy: number;
  };
}

export interface ModelResult {
  modelId: string;
  modelType: string;
  metrics: {
    accuracy?: number;
    precision?: number;
    recall?: number;
    f1Score?: number;
    r2Score?: number;
    mse?: number;
    mae?: number;
  };
  status: string;
  createdAt: string;
}

export interface TrainingRequest {
  modelType: string;
  targetColumn: string;
  featureColumns: string[];
  hyperparameters?: Record<string, any>;
  testSize?: number;
}

export interface PredictionRequest {
  modelId: string;
  features: Record<string, any>;
}

export interface PredictionResponse {
  prediction: any;
  probability?: number[];
  modelId: string;
  timestamp: string;
}

export interface DataSummary {
  totalRecords: number;
  columns: string[];
  dataTypes: Record<string, string>;
  missingValues: Record<string, number>;
  uniqueValues: Record<string, number>;
  dateRange: {
    minYear: string;
    maxYear: string;
  };
  vehicleTypes: Record<string, number>;
}

export interface VisualizationData {
  chartType: string;
  data: Record<string, any>;
  metadata: Record<string, any>;
}

class ApiService {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        // Add auth token if available
        const token = localStorage.getItem('auth_token');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response?.status === 401) {
          // Handle unauthorized access
          localStorage.removeItem('auth_token');
          window.location.href = '/login';
        }
        return Promise.reject(error);
      }
    );
  }

  // Health endpoints
  async getHealthStatus(): Promise<any> {
    const response = await this.client.get('/health');
    return response.data;
  }

  async getDetailedHealth(): Promise<any> {
    const response = await this.client.get('/health/detailed');
    return response.data;
  }

  // Data endpoints
  async getDataSummary(): Promise<DataSummary> {
    const response = await this.client.get('/api/v1/data/summary');
    return response.data;
  }

  async getVisualizationData(chartType: string, limit?: number): Promise<VisualizationData> {
    const params = limit ? { limit } : {};
    const response = await this.client.get(`/api/v1/data/visualization/${chartType}`, { params });
    return response.data;
  }

  async uploadDataset(file: File): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await this.client.post('/api/v1/data/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  async exportData(format: string, filters?: Record<string, any>): Promise<any> {
    const params = filters ? { filters: JSON.stringify(filters) } : {};
    const response = await this.client.get(`/api/v1/data/export/${format}`, { params });
    return response.data;
  }

  async getDataStatistics(): Promise<any> {
    const response = await this.client.get('/api/v1/data/statistics');
    return response.data;
  }

  // Model endpoints
  async trainModel(request: TrainingRequest): Promise<ModelResult> {
    const response = await this.client.post('/api/v1/models/train', request);
    return response.data;
  }

  async getModels(): Promise<{ models: ModelResult[] }> {
    const response = await this.client.get('/api/v1/models/list');
    return response.data;
  }

  async getModelMetrics(modelId: string): Promise<any> {
    const response = await this.client.get(`/api/v1/models/${modelId}/metrics`);
    return response.data;
  }

  async makePrediction(request: PredictionRequest): Promise<PredictionResponse> {
    const response = await this.client.post('/api/v1/models/predict', request);
    return response.data;
  }

  async deleteModel(modelId: string): Promise<any> {
    const response = await this.client.delete(`/api/v1/models/${modelId}`);
    return response.data;
  }

  // Batch operations
  async batchPredict(modelId: string, data: Record<string, any>[]): Promise<any> {
    const response = await this.client.post('/api/v1/models/batch-predict', {
      modelId,
      data,
    });
    return response.data;
  }

  // Model comparison
  async compareModels(modelIds: string[]): Promise<any> {
    const response = await this.client.post('/api/v1/models/compare', { modelIds });
    return response.data;
  }

  // Feature importance
  async getFeatureImportance(modelId: string): Promise<any> {
    const response = await this.client.get(`/api/v1/models/${modelId}/feature-importance`);
    return response.data;
  }

  // Data validation
  async validateData(data: any): Promise<any> {
    const response = await this.client.post('/api/v1/data/validate', data);
    return response.data;
  }

  // A/B testing endpoints
  async createExperiment(config: any): Promise<any> {
    const response = await this.client.post('/api/v1/experiments', config);
    return response.data;
  }

  async getExperiments(): Promise<any> {
    const response = await this.client.get('/api/v1/experiments');
    return response.data;
  }
}

// Export singleton instance
export const apiService = new ApiService();
export default apiService;