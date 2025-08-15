import React, { useState, useEffect } from 'react';
import {
  Card,
  Form,
  Input,
  Button,
  Select,
  Upload,
  Table,
  Row,
  Col,
  Statistic,
  Alert,
  Space,
  Tag,
  Progress,
  Divider,
  Typography,
  InputNumber,
  DatePicker,
  Result,
  Spin,
  message
} from 'antd';
import {
  UploadOutlined,
  SendOutlined,
  DownloadOutlined,
  ThunderboltOutlined,
  FileExcelOutlined,
  CheckCircleOutlined,
  InfoCircleOutlined
} from '@ant-design/icons';
import { UploadFile } from 'antd/es/upload/interface';
import { modelAPI, dataAPI } from '../services/api';
import { Line, Scatter } from 'react-chartjs-2';

const { Title, Text, Paragraph } = Typography;
const { Option } = Select;
const { Dragger } = Upload;

interface PredictionResult {
  id: string;
  input_data: Record<string, any>;
  prediction: any;
  confidence?: number;
  model_used: string;
  timestamp: string;
  processing_time: number;
}

interface BatchPredictionJob {
  job_id: string;
  status: 'processing' | 'completed' | 'failed';
  total_records: number;
  processed_records: number;
  predictions?: any[];
  error?: string;
}

export const Predictions: React.FC = () => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [availableModels, setAvailableModels] = useState<any[]>([]);
  const [batchJob, setBatchJob] = useState<BatchPredictionJob | null>(null);
  const [fileList, setFileList] = useState<UploadFile[]>([]);
  const [predictionMode, setPredictionMode] = useState<'single' | 'batch'>('single');

  useEffect(() => {
    fetchAvailableModels();
    loadPredictionHistory();
  }, []);

  const fetchAvailableModels = async () => {
    try {
      const response = await modelAPI.getAvailableModels();
      setAvailableModels(response.data);
      if (response.data.length > 0) {
        setSelectedModel(response.data[0].model_id);
      }
    } catch (error) {
      message.error('Failed to load available models');
    }
  };

  const loadPredictionHistory = async () => {
    try {
      const response = await modelAPI.getPredictionHistory();
      setPredictions(response.data);
    } catch (error) {
      console.error('Failed to load prediction history:', error);
    }
  };

  const handleSinglePrediction = async (values: any) => {
    setLoading(true);
    
    const predictionData = {
      model_id: selectedModel,
      features: {
        county: values.county,
        city: values.city,
        state: values.state || 'WA',
        postal_code: values.postal_code,
        model_year: values.model_year,
        make: values.make,
        model: values.model,
        ev_type: values.ev_type,
        cafv_eligibility: values.cafv_eligibility,
        electric_range: values.electric_range,
        base_msrp: values.base_msrp,
        legislative_district: values.legislative_district,
        electric_utility: values.electric_utility
      }
    };

    try {
      const response = await modelAPI.makePrediction(predictionData);
      
      const newPrediction: PredictionResult = {
        id: `pred_${Date.now()}`,
        input_data: predictionData.features,
        prediction: response.data.prediction,
        confidence: response.data.confidence,
        model_used: selectedModel,
        timestamp: new Date().toISOString(),
        processing_time: response.data.processing_time || 0.5
      };

      setPredictions(prev => [newPrediction, ...prev]);
      
      message.success({
        content: `Prediction completed! Result: ${response.data.prediction}`,
        icon: <CheckCircleOutlined />
      });

      form.resetFields();
    } catch (error) {
      message.error('Prediction failed. Please check your input and try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleBatchPrediction = async () => {
    if (fileList.length === 0) {
      message.warning('Please upload a CSV file for batch prediction');
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append('file', fileList[0] as any);
    formData.append('model_id', selectedModel);

    try {
      const response = await modelAPI.batchPredict(formData);
      
      setBatchJob({
        job_id: response.data.job_id,
        status: 'processing',
        total_records: response.data.total_records,
        processed_records: 0
      });

      // Poll for batch job status
      const pollInterval = setInterval(async () => {
        const statusResponse = await modelAPI.getBatchJobStatus(response.data.job_id);
        
        setBatchJob(statusResponse.data);
        
        if (statusResponse.data.status === 'completed') {
          clearInterval(pollInterval);
          message.success('Batch prediction completed successfully!');
          setLoading(false);
        } else if (statusResponse.data.status === 'failed') {
          clearInterval(pollInterval);
          message.error('Batch prediction failed: ' + statusResponse.data.error);
          setLoading(false);
        }
      }, 2000);

    } catch (error) {
      message.error('Failed to start batch prediction');
      setLoading(false);
    }
  };

  const downloadPredictions = () => {
    const csvContent = predictions.map(p => ({
      timestamp: p.timestamp,
      model: p.model_used,
      prediction: p.prediction,
      confidence: p.confidence,
      ...p.input_data
    }));

    const csv = convertToCSV(csvContent);
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `predictions_${Date.now()}.csv`;
    a.click();
  };

  const convertToCSV = (data: any[]) => {
    if (data.length === 0) return '';
    const headers = Object.keys(data[0]);
    const csvHeaders = headers.join(',');
    const csvRows = data.map(row => 
      headers.map(header => JSON.stringify(row[header] || '')).join(',')
    );
    return [csvHeaders, ...csvRows].join('\n');
  };

  const uploadProps = {
    onRemove: (file: UploadFile) => {
      setFileList([]);
    },
    beforeUpload: (file: UploadFile) => {
      setFileList([file]);
      return false;
    },
    fileList,
  };

  const predictionColumns = [
    {
      title: 'Time',
      dataIndex: 'timestamp',
      key: 'timestamp',
      render: (time: string) => new Date(time).toLocaleString(),
      width: 180
    },
    {
      title: 'Model',
      dataIndex: 'model_used',
      key: 'model_used',
      render: (model: string) => <Tag color="blue">{model}</Tag>,
      width: 150
    },
    {
      title: 'Prediction',
      dataIndex: 'prediction',
      key: 'prediction',
      render: (pred: any) => (
        <Tag color="green" style={{ fontSize: '14px' }}>
          {typeof pred === 'object' ? JSON.stringify(pred) : pred}
        </Tag>
      ),
      width: 150
    },
    {
      title: 'Confidence',
      dataIndex: 'confidence',
      key: 'confidence',
      render: (conf: number) => conf ? (
        <Progress
          percent={Math.round(conf * 100)}
          size="small"
          status={conf > 0.8 ? 'success' : conf > 0.6 ? 'normal' : 'exception'}
        />
      ) : '-',
      width: 150
    },
    {
      title: 'Processing Time',
      dataIndex: 'processing_time',
      key: 'processing_time',
      render: (time: number) => `${time.toFixed(3)}s`,
      width: 120
    }
  ];

  return (
    <div className="predictions-container">
      <Title level={2}>
        <ThunderboltOutlined /> Model Predictions
      </Title>

      <Row gutter={[16, 16]}>
        <Col span={24}>
          <Card>
            <Row gutter={16}>
              <Col span={6}>
                <Statistic
                  title="Total Predictions"
                  value={predictions.length}
                  prefix={<CheckCircleOutlined />}
                />
              </Col>
              <Col span={6}>
                <Statistic
                  title="Available Models"
                  value={availableModels.length}
                  valueStyle={{ color: '#3f8600' }}
                />
              </Col>
              <Col span={6}>
                <Statistic
                  title="Avg. Processing Time"
                  value={
                    predictions.length > 0
                      ? (predictions.reduce((acc, p) => acc + p.processing_time, 0) / predictions.length).toFixed(3)
                      : 0
                  }
                  suffix="s"
                  valueStyle={{ color: '#1890ff' }}
                />
              </Col>
              <Col span={6}>
                <Statistic
                  title="Avg. Confidence"
                  value={
                    predictions.filter(p => p.confidence).length > 0
                      ? Math.round(
                          (predictions.filter(p => p.confidence).reduce((acc, p) => acc + (p.confidence || 0), 0) /
                            predictions.filter(p => p.confidence).length) * 100
                        )
                      : 0
                  }
                  suffix="%"
                  valueStyle={{ color: '#722ed1' }}
                />
              </Col>
            </Row>
          </Card>
        </Col>

        <Col xs={24} lg={14}>
          <Card
            title="Make Predictions"
            extra={
              <Select
                value={predictionMode}
                onChange={setPredictionMode}
                style={{ width: 120 }}
              >
                <Option value="single">Single</Option>
                <Option value="batch">Batch</Option>
              </Select>
            }
          >
            {predictionMode === 'single' ? (
              <Form
                form={form}
                layout="vertical"
                onFinish={handleSinglePrediction}
              >
                <Row gutter={16}>
                  <Col span={24}>
                    <Form.Item
                      label="Select Model"
                      required
                    >
                      <Select
                        value={selectedModel}
                        onChange={setSelectedModel}
                        placeholder="Choose a trained model"
                        size="large"
                      >
                        {availableModels.map(model => (
                          <Option key={model.model_id} value={model.model_id}>
                            <Space>
                              {model.name}
                              <Tag color="cyan">{model.type}</Tag>
                              <Text type="secondary">
                                Accuracy: {(model.accuracy * 100).toFixed(1)}%
                              </Text>
                            </Space>
                          </Option>
                        ))}
                      </Select>
                    </Form.Item>
                  </Col>
                </Row>

                <Divider>Vehicle Information</Divider>

                <Row gutter={16}>
                  <Col span={8}>
                    <Form.Item
                      name="make"
                      label="Make"
                      rules={[{ required: true, message: 'Required' }]}
                    >
                      <Input placeholder="e.g., TESLA" />
                    </Form.Item>
                  </Col>
                  <Col span={8}>
                    <Form.Item
                      name="model"
                      label="Model"
                      rules={[{ required: true, message: 'Required' }]}
                    >
                      <Input placeholder="e.g., Model 3" />
                    </Form.Item>
                  </Col>
                  <Col span={8}>
                    <Form.Item
                      name="model_year"
                      label="Model Year"
                      rules={[{ required: true, message: 'Required' }]}
                    >
                      <InputNumber
                        min={2010}
                        max={2025}
                        style={{ width: '100%' }}
                        placeholder="2024"
                      />
                    </Form.Item>
                  </Col>
                </Row>

                <Row gutter={16}>
                  <Col span={8}>
                    <Form.Item
                      name="ev_type"
                      label="EV Type"
                      rules={[{ required: true, message: 'Required' }]}
                    >
                      <Select placeholder="Select EV type">
                        <Option value="BEV">Battery Electric (BEV)</Option>
                        <Option value="PHEV">Plug-in Hybrid (PHEV)</Option>
                      </Select>
                    </Form.Item>
                  </Col>
                  <Col span={8}>
                    <Form.Item
                      name="electric_range"
                      label="Electric Range (miles)"
                      rules={[{ required: true, message: 'Required' }]}
                    >
                      <InputNumber
                        min={0}
                        max={500}
                        style={{ width: '100%' }}
                        placeholder="300"
                      />
                    </Form.Item>
                  </Col>
                  <Col span={8}>
                    <Form.Item
                      name="base_msrp"
                      label="Base MSRP ($)"
                    >
                      <InputNumber
                        min={0}
                        max={200000}
                        style={{ width: '100%' }}
                        placeholder="45000"
                      />
                    </Form.Item>
                  </Col>
                </Row>

                <Divider>Location Information</Divider>

                <Row gutter={16}>
                  <Col span={8}>
                    <Form.Item
                      name="city"
                      label="City"
                      rules={[{ required: true, message: 'Required' }]}
                    >
                      <Input placeholder="e.g., Seattle" />
                    </Form.Item>
                  </Col>
                  <Col span={8}>
                    <Form.Item
                      name="county"
                      label="County"
                    >
                      <Input placeholder="e.g., King" />
                    </Form.Item>
                  </Col>
                  <Col span={8}>
                    <Form.Item
                      name="postal_code"
                      label="Postal Code"
                      rules={[{ required: true, message: 'Required' }]}
                    >
                      <Input placeholder="98101" />
                    </Form.Item>
                  </Col>
                </Row>

                <Row gutter={16}>
                  <Col span={12}>
                    <Form.Item
                      name="cafv_eligibility"
                      label="CAFV Eligibility"
                    >
                      <Select placeholder="Select eligibility">
                        <Option value="Clean Alternative Fuel Vehicle Eligible">Eligible</Option>
                        <Option value="Not Eligible">Not Eligible</Option>
                        <Option value="Unknown">Unknown</Option>
                      </Select>
                    </Form.Item>
                  </Col>
                  <Col span={12}>
                    <Form.Item
                      name="electric_utility"
                      label="Electric Utility"
                    >
                      <Input placeholder="e.g., SEATTLE CITY LIGHT" />
                    </Form.Item>
                  </Col>
                </Row>

                <Form.Item>
                  <Button
                    type="primary"
                    htmlType="submit"
                    loading={loading}
                    icon={<SendOutlined />}
                    size="large"
                    block
                  >
                    Make Prediction
                  </Button>
                </Form.Item>
              </Form>
            ) : (
              <div>
                <Alert
                  message="Batch Prediction"
                  description="Upload a CSV file with multiple records for batch prediction. The file should contain the same fields as the single prediction form."
                  type="info"
                  showIcon
                  icon={<InfoCircleOutlined />}
                  style={{ marginBottom: 16 }}
                />

                <Dragger {...uploadProps}>
                  <p className="ant-upload-drag-icon">
                    <FileExcelOutlined style={{ fontSize: 48, color: '#1890ff' }} />
                  </p>
                  <p className="ant-upload-text">
                    Click or drag CSV file to this area to upload
                  </p>
                  <p className="ant-upload-hint">
                    Support for single file upload. File should contain EV data in CSV format.
                  </p>
                </Dragger>

                {batchJob && (
                  <Card style={{ marginTop: 16 }}>
                    <Progress
                      percent={Math.round((batchJob.processed_records / batchJob.total_records) * 100)}
                      status={batchJob.status === 'completed' ? 'success' : 'active'}
                    />
                    <Text>
                      Processing: {batchJob.processed_records} / {batchJob.total_records} records
                    </Text>
                  </Card>
                )}

                <Button
                  type="primary"
                  onClick={handleBatchPrediction}
                  loading={loading}
                  disabled={fileList.length === 0}
                  icon={<UploadOutlined />}
                  size="large"
                  block
                  style={{ marginTop: 16 }}
                >
                  Start Batch Prediction
                </Button>
              </div>
            )}
          </Card>
        </Col>

        <Col xs={24} lg={10}>
          <Card
            title="Prediction Confidence Distribution"
            bodyStyle={{ height: 400 }}
          >
            {predictions.filter(p => p.confidence).length > 0 ? (
              <Scatter
                data={{
                  datasets: [{
                    label: 'Prediction Confidence',
                    data: predictions
                      .filter(p => p.confidence)
                      .map((p, index) => ({
                        x: index,
                        y: p.confidence || 0
                      })),
                    backgroundColor: 'rgba(75, 192, 192, 0.6)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1
                  }]
                }}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: {
                    legend: { display: false },
                    title: {
                      display: true,
                      text: 'Confidence Scores Over Time'
                    }
                  },
                  scales: {
                    y: {
                      beginAtZero: true,
                      max: 1,
                      ticks: {
                        callback: function(value: any) {
                          return (value * 100) + '%';
                        }
                      }
                    },
                    x: {
                      title: {
                        display: true,
                        text: 'Prediction Index'
                      }
                    }
                  }
                }}
              />
            ) : (
              <Result
                icon={<InfoCircleOutlined />}
                title="No confidence data available"
                subTitle="Make predictions to see confidence distribution"
              />
            )}
          </Card>
        </Col>

        <Col span={24}>
          <Card
            title="Prediction History"
            extra={
              <Button
                icon={<DownloadOutlined />}
                onClick={downloadPredictions}
                disabled={predictions.length === 0}
              >
                Export CSV
              </Button>
            }
          >
            <Table
              columns={predictionColumns}
              dataSource={predictions}
              rowKey="id"
              pagination={{ pageSize: 10 }}
              scroll={{ x: 1000 }}
              locale={{ emptyText: 'No predictions yet. Make your first prediction above!' }}
            />
          </Card>
        </Col>
      </Row>
    </div>
  );
};