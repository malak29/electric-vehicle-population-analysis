import React, { useState, useEffect } from 'react';
import {
  Card,
  Form,
  Select,
  Button,
  Table,
  Progress,
  Alert,
  Row,
  Col,
  Statistic,
  Space,
  notification,
  Tabs,
  InputNumber,
  Switch,
  Divider,
  Tag,
  Typography
} from 'antd';
import {
  PlayCircleOutlined,
  ReloadOutlined,
  BarChartOutlined,
  RocketOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined
} from '@ant-design/icons';
import { Line, Bar } from 'react-chartjs-2';
import { modelAPI } from '../services/api';

const { Option } = Select;
const { Title, Text } = Typography;
const { TabPane } = Tabs;

interface ModelMetrics {
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
  rmse?: number;
  r2?: number;
  mse?: number;
  mae?: number;
}

interface TrainingJob {
  id: string;
  model_type: string;
  status: 'pending' | 'training' | 'completed' | 'failed';
  progress: number;
  metrics?: ModelMetrics;
  created_at: string;
  duration?: number;
}

export const ModelTraining: React.FC = () => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [trainingJobs, setTrainingJobs] = useState<TrainingJob[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('random_forest');
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [experimentHistory, setExperimentHistory] = useState<any[]>([]);

  useEffect(() => {
    fetchExperimentHistory();
    // Poll for active training jobs
    const interval = setInterval(checkTrainingStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchExperimentHistory = async () => {
    try {
      const response = await modelAPI.getExperiments();
      setExperimentHistory(response.data);
    } catch (error) {
      console.error('Failed to fetch experiments:', error);
    }
  };

  const checkTrainingStatus = async () => {
    const activeJobs = trainingJobs.filter(job => job.status === 'training');
    for (const job of activeJobs) {
      try {
        const response = await modelAPI.getJobStatus(job.id);
        updateJobStatus(job.id, response.data);
      } catch (error) {
        console.error('Failed to check job status:', error);
      }
    }
  };

  const updateJobStatus = (jobId: string, statusData: any) => {
    setTrainingJobs(prev => prev.map(job => 
      job.id === jobId 
        ? { ...job, ...statusData }
        : job
    ));

    if (statusData.status === 'completed') {
      setMetrics(statusData.metrics);
      notification.success({
        message: 'Training Completed',
        description: `Model ${jobId} has been successfully trained!`,
        icon: <CheckCircleOutlined style={{ color: '#52c41a' }} />
      });
    }
  };

  const handleTrainModel = async (values: any) => {
    setLoading(true);
    setTrainingProgress(0);

    const trainingConfig = {
      model_type: values.model_type,
      parameters: {
        test_size: values.test_size / 100,
        cv_folds: values.cv_folds,
        max_depth: values.max_depth,
        n_estimators: values.n_estimators,
        learning_rate: values.learning_rate,
        enable_grid_search: values.enable_grid_search
      }
    };

    try {
      // Start training
      const response = await modelAPI.trainModel(trainingConfig);
      const newJob: TrainingJob = {
        id: response.data.job_id,
        model_type: values.model_type,
        status: 'training',
        progress: 0,
        created_at: new Date().toISOString()
      };

      setTrainingJobs(prev => [newJob, ...prev]);

      // Simulate progress updates
      const progressInterval = setInterval(() => {
        setTrainingProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 100;
          }
          return prev + Math.random() * 15;
        });
      }, 1000);

      // Wait for completion
      setTimeout(async () => {
        const metricsResponse = await modelAPI.getModelMetrics(response.data.job_id);
        setMetrics(metricsResponse.data);
        updateJobStatus(response.data.job_id, {
          status: 'completed',
          progress: 100,
          metrics: metricsResponse.data,
          duration: Math.floor(Math.random() * 60) + 30
        });
        setTrainingProgress(100);
        fetchExperimentHistory();
      }, 5000);

    } catch (error) {
      notification.error({
        message: 'Training Failed',
        description: 'Failed to start model training. Please try again.',
        icon: <CloseCircleOutlined style={{ color: '#ff4d4f' }} />
      });
    } finally {
      setLoading(false);
    }
  };

  const modelOptions = [
    { value: 'random_forest', label: 'Random Forest', type: 'classification' },
    { value: 'gradient_boosting', label: 'Gradient Boosting', type: 'classification' },
    { value: 'decision_tree', label: 'Decision Tree', type: 'classification' },
    { value: 'logistic_regression', label: 'Logistic Regression', type: 'classification' },
    { value: 'linear_regression', label: 'Linear Regression', type: 'regression' },
    { value: 'xgboost', label: 'XGBoost', type: 'classification' },
    { value: 'neural_network', label: 'Neural Network', type: 'both' },
    { value: 'k_means', label: 'K-Means Clustering', type: 'clustering' }
  ];

  const columns = [
    {
      title: 'Model',
      dataIndex: 'model_type',
      key: 'model_type',
      render: (text: string) => <Tag color="blue">{text.toUpperCase()}</Tag>
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const colors: any = {
          pending: 'orange',
          training: 'processing',
          completed: 'success',
          failed: 'error'
        };
        return <Tag color={colors[status]}>{status.toUpperCase()}</Tag>;
      }
    },
    {
      title: 'Progress',
      dataIndex: 'progress',
      key: 'progress',
      render: (progress: number) => <Progress percent={progress} size="small" />
    },
    {
      title: 'Accuracy',
      dataIndex: 'metrics',
      key: 'accuracy',
      render: (metrics: ModelMetrics) => 
        metrics?.accuracy ? `${(metrics.accuracy * 100).toFixed(2)}%` : '-'
    },
    {
      title: 'Duration',
      dataIndex: 'duration',
      key: 'duration',
      render: (duration: number) => duration ? `${duration}s` : '-'
    }
  ];

  const metricsChartData = {
    labels: ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    datasets: [
      {
        label: 'Model Performance',
        data: metrics ? [
          metrics.accuracy || 0,
          metrics.precision || 0,
          metrics.recall || 0,
          metrics.f1_score || 0
        ] : [0, 0, 0, 0],
        backgroundColor: [
          'rgba(54, 162, 235, 0.8)',
          'rgba(75, 192, 192, 0.8)',
          'rgba(255, 206, 86, 0.8)',
          'rgba(153, 102, 255, 0.8)'
        ],
        borderColor: [
          'rgba(54, 162, 235, 1)',
          'rgba(75, 192, 192, 1)',
          'rgba(255, 206, 86, 1)',
          'rgba(153, 102, 255, 1)'
        ],
        borderWidth: 1
      }
    ]
  };

  return (
    <div className="model-training-container">
      <Title level={2}>
        <RocketOutlined /> Model Training Center
      </Title>
      
      <Row gutter={[16, 16]}>
        <Col span={24}>
          {metrics && (
            <Card className="metrics-summary">
              <Row gutter={16}>
                <Col span={6}>
                  <Statistic
                    title="Best Accuracy"
                    value={(metrics.accuracy || 0) * 100}
                    precision={2}
                    suffix="%"
                    valueStyle={{ color: '#3f8600' }}
                  />
                </Col>
                <Col span={6}>
                  <Statistic
                    title="Precision"
                    value={(metrics.precision || 0) * 100}
                    precision={2}
                    suffix="%"
                    valueStyle={{ color: '#1890ff' }}
                  />
                </Col>
                <Col span={6}>
                  <Statistic
                    title="Recall"
                    value={(metrics.recall || 0) * 100}
                    precision={2}
                    suffix="%"
                    valueStyle={{ color: '#722ed1' }}
                  />
                </Col>
                <Col span={6}>
                  <Statistic
                    title="F1-Score"
                    value={(metrics.f1_score || 0) * 100}
                    precision={2}
                    suffix="%"
                    valueStyle={{ color: '#fa8c16' }}
                  />
                </Col>
              </Row>
            </Card>
          )}
        </Col>

        <Col xs={24} lg={12}>
          <Card title="Configure Training" extra={<BarChartOutlined />}>
            <Form
              form={form}
              layout="vertical"
              onFinish={handleTrainModel}
              initialValues={{
                model_type: 'random_forest',
                test_size: 20,
                cv_folds: 5,
                n_estimators: 100,
                max_depth: 10,
                learning_rate: 0.01,
                enable_grid_search: false
              }}
            >
              <Form.Item
                name="model_type"
                label="Select Model"
                rules={[{ required: true, message: 'Please select a model!' }]}
              >
                <Select
                  size="large"
                  onChange={setSelectedModel}
                  showSearch
                  placeholder="Choose ML model"
                >
                  {modelOptions.map(model => (
                    <Option key={model.value} value={model.value}>
                      <Space>
                        {model.label}
                        <Tag color={
                          model.type === 'classification' ? 'blue' :
                          model.type === 'regression' ? 'green' : 'purple'
                        }>
                          {model.type}
                        </Tag>
                      </Space>
                    </Option>
                  ))}
                </Select>
              </Form.Item>

              <Row gutter={16}>
                <Col span={12}>
                  <Form.Item
                    name="test_size"
                    label="Test Size (%)"
                    tooltip="Percentage of data for testing"
                  >
                    <InputNumber
                      min={10}
                      max={40}
                      style={{ width: '100%' }}
                      formatter={value => `${value}%`}
                      parser={value => value!.replace('%', '')}
                    />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item
                    name="cv_folds"
                    label="Cross-Validation Folds"
                    tooltip="Number of folds for cross-validation"
                  >
                    <InputNumber min={2} max={10} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
              </Row>

              {['random_forest', 'gradient_boosting', 'xgboost'].includes(selectedModel) && (
                <Row gutter={16}>
                  <Col span={12}>
                    <Form.Item name="n_estimators" label="Number of Estimators">
                      <InputNumber min={10} max={500} style={{ width: '100%' }} />
                    </Form.Item>
                  </Col>
                  <Col span={12}>
                    <Form.Item name="max_depth" label="Max Depth">
                      <InputNumber min={1} max={50} style={{ width: '100%' }} />
                    </Form.Item>
                  </Col>
                </Row>
              )}

              {['gradient_boosting', 'xgboost'].includes(selectedModel) && (
                <Form.Item name="learning_rate" label="Learning Rate">
                  <InputNumber
                    min={0.001}
                    max={1}
                    step={0.01}
                    style={{ width: '100%' }}
                  />
                </Form.Item>
              )}

              <Form.Item name="enable_grid_search" valuePropName="checked">
                <Switch /> Enable Grid Search for Hyperparameter Tuning
              </Form.Item>

              <Form.Item>
                <Button
                  type="primary"
                  htmlType="submit"
                  loading={loading}
                  icon={<PlayCircleOutlined />}
                  size="large"
                  block
                >
                  Start Training
                </Button>
              </Form.Item>
            </Form>

            {loading && (
              <Progress
                percent={trainingProgress}
                status="active"
                strokeColor={{
                  '0%': '#108ee9',
                  '100%': '#87d068',
                }}
              />
            )}
          </Card>
        </Col>

        <Col xs={24} lg={12}>
          <Card title="Performance Metrics" extra={<ReloadOutlined onClick={fetchExperimentHistory} />}>
            {metrics ? (
              <Bar
                data={metricsChartData}
                options={{
                  responsive: true,
                  plugins: {
                    legend: { display: false },
                    title: {
                      display: true,
                      text: 'Model Performance Metrics'
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
                    }
                  }
                }}
              />
            ) : (
              <div style={{ textAlign: 'center', padding: '40px' }}>
                <Text type="secondary">Train a model to see performance metrics</Text>
              </div>
            )}
          </Card>
        </Col>

        <Col span={24}>
          <Card title="Training History">
            <Table
              columns={columns}
              dataSource={trainingJobs}
              rowKey="id"
              pagination={{ pageSize: 5 }}
              locale={{ emptyText: 'No training jobs yet' }}
            />
          </Card>
        </Col>

        <Col span={24}>
          <Card title="Experiment Comparison">
            <Tabs defaultActiveKey="1">
              <TabPane tab="Accuracy Comparison" key="1">
                <Line
                  data={{
                    labels: experimentHistory.map((_, i) => `Exp ${i + 1}`),
                    datasets: [{
                      label: 'Accuracy Trend',
                      data: experimentHistory.map(exp => exp.accuracy || 0),
                      borderColor: 'rgb(75, 192, 192)',
                      backgroundColor: 'rgba(75, 192, 192, 0.2)',
                      tension: 0.1
                    }]
                  }}
                  options={{
                    responsive: true,
                    plugins: {
                      legend: { position: 'top' as const },
                      title: {
                        display: true,
                        text: 'Model Accuracy Over Experiments'
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
                      }
                    }
                  }}
                />
              </TabPane>
              <TabPane tab="Model Comparison" key="2">
                <Table
                  dataSource={experimentHistory}
                  columns={[
                    { title: 'Model', dataIndex: 'model_name', key: 'model_name' },
                    { 
                      title: 'Accuracy', 
                      dataIndex: 'accuracy', 
                      key: 'accuracy',
                      render: (val: number) => `${(val * 100).toFixed(2)}%`,
                      sorter: (a: any, b: any) => a.accuracy - b.accuracy
                    },
                    { 
                      title: 'Training Time', 
                      dataIndex: 'training_time', 
                      key: 'training_time',
                      render: (val: number) => `${val}s`
                    },
                    { 
                      title: 'Parameters', 
                      dataIndex: 'parameters', 
                      key: 'parameters',
                      render: (params: any) => JSON.stringify(params)
                    }
                  ]}
                  rowKey="id"
                />
              </TabPane>
            </Tabs>
          </Card>
        </Col>
      </Row>
    </div>
  );
};