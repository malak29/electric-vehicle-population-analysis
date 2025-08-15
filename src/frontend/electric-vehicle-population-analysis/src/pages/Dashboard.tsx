import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Spin,
  Alert,
  Tabs,
  Button,
  Space,
  Typography,
  Progress,
  Tag,
} from 'antd';
import {
  CarOutlined,
  ThunderboltOutlined,
  BarChartOutlined,
  ReloadOutlined,
  TrophyOutlined,
  ExperimentOutlined,
} from '@ant-design/icons';
import { Line, Column, Pie, Area } from '@ant-design/plots';

import { apiService } from '../services/api';
import type { DataSummary, VisualizationData, ModelResult } from '../services/api';

const { Title, Text } = Typography;
const { TabPane } = Tabs;

interface DashboardStats {
  totalVehicles: number;
  bevCount: number;
  phevCount: number;
  averageRange: number;
  topMake: string;
  growthRate: number;
}

const Dashboard: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [chartData, setChartData] = useState<{
    vehicleTypes: any[];
    rangeByYear: any[];
    makeDistribution: any[];
    geographicData: any[];
  } | null>(null);
  const [models, setModels] = useState<ModelResult[]>([]);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Load all data in parallel
      const [summary, vehicleTypes, rangeByYear, makeDistribution, geographic, modelList] = 
        await Promise.all([
          apiService.getDataSummary(),
          apiService.getVisualizationData('vehicle_type_distribution'),
          apiService.getVisualizationData('range_by_year'),
          apiService.getVisualizationData('make_distribution', 10),
          apiService.getVisualizationData('geographic_distribution', 20),
          apiService.getModels(),
        ]);

      // Process stats
      const processedStats: DashboardStats = {
        totalVehicles: summary.totalRecords,
        bevCount: summary.vehicleTypes['Battery Electric Vehicle (BEV)'] || 0,
        phevCount: summary.vehicleTypes['Plug-in Hybrid Electric Vehicle (PHEV)'] || 0,
        averageRange: Object.values(rangeByYear.data.ranges as number[]).reduce((a, b) => a + b, 0) / 
                     Object.keys(rangeByYear.data.ranges).length,
        topMake: makeDistribution.data.makes[0],
        growthRate: calculateGrowthRate(rangeByYear.data),
      };

      setStats(processedStats);
      
      // Process chart data
      setChartData({
        vehicleTypes: vehicleTypes.data.labels.map((label: string, index: number) => ({
          type: label,
          count: vehicleTypes.data.values[index],
        })),
        rangeByYear: Object.entries(rangeByYear.data.years).map(([year, range]) => ({
          year: parseInt(year),
          range: range,
        })),
        makeDistribution: makeDistribution.data.makes.map((make: string, index: number) => ({
          make,
          count: makeDistribution.data.counts[index],
        })),
        geographicData: geographic.data,
      });

      setModels(modelList.models);
      
    } catch (err) {
      setError('Failed to load dashboard data');
      console.error('Dashboard data loading error:', err);
    } finally {
      setLoading(false);
    }
  };

  const calculateGrowthRate = (rangeData: any): number => {
    const years = Object.keys(rangeData.ranges).map(Number).sort();
    if (years.length < 2) return 0;
    
    const firstYear = rangeData.ranges[years[0]];
    const lastYear = rangeData.ranges[years[years.length - 1]];
    
    return ((lastYear - firstYear) / firstYear) * 100;
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    await loadDashboardData();
    setRefreshing(false);
  };

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '50px' }}>
        <Spin size="large" />
        <div style={{ marginTop: 16 }}>Loading dashboard data...</div>
      </div>
    );
  }

  if (error) {
    return (
      <Alert
        message="Error"
        description={error}
        type="error"
        showIcon
        action={
          <Button size="small" danger onClick={loadDashboardData}>
            Retry
          </Button>
        }
      />
    );
  }

  return (
    <div>
      {/* Header */}
      <Row justify="space-between" align="middle" style={{ marginBottom: 24 }}>
        <Col>
          <Title level={2} style={{ margin: 0 }}>
            <CarOutlined /> EV Population Dashboard
          </Title>
          <Text type="secondary">
            Real-time insights into electric vehicle adoption patterns
          </Text>
        </Col>
        <Col>
          <Button
            icon={<ReloadOutlined />}
            onClick={handleRefresh}
            loading={refreshing}
          >
            Refresh
          </Button>
        </Col>
      </Row>

      {/* Key Statistics */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Total EVs"
              value={stats?.totalVehicles}
              prefix={<CarOutlined />}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Battery Electric"
              value={stats?.bevCount}
              prefix={<ThunderboltOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Plug-in Hybrid"
              value={stats?.phevCount}
              prefix={<ThunderboltOutlined />}
              valueStyle={{ color: '#722ed1' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Avg Range"
              value={stats?.averageRange?.toFixed(0)}
              suffix="miles"
              prefix={<BarChartOutlined />}
              valueStyle={{ color: '#fa8c16' }}
            />
          </Card>
        </Col>
      </Row>

      {/* Charts */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} lg={12}>
          <Card title="Vehicle Type Distribution" extra={<Tag color="blue">Live Data</Tag>}>
            {chartData && (
              <Pie
                data={chartData.vehicleTypes}
                angleField="count"
                colorField="type"
                radius={0.8}
                label={{
                  type: 'outer',
                  content: '{name} {percentage}',
                }}
                interactions={[{ type: 'element-selected' }, { type: 'element-active' }]}
                height={300}
              />
            )}
          </Card>
        </Col>
        <Col xs={24} lg={12}>
          <Card title="Electric Range by Year" extra={<Tag color="green">Trending</Tag>}>
            {chartData && (
              <Line
                data={chartData.rangeByYear}
                xField="year"
                yField="range"
                height={300}
                point={{
                  size: 5,
                  shape: 'diamond',
                  style: {
                    fill: 'white',
                    stroke: '#1890ff',
                    lineWidth: 2,
                  },
                }}
                tooltip={{
                  showMarkers: false,
                }}
                state={{
                  active: {
                    style: {
                      shadowBlur: 4,
                      stroke: '#000',
                      fill: 'red',
                    },
                  },
                }}
              />
            )}
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]}>
        <Col xs={24} lg={16}>
          <Card title="Top Manufacturers" extra={<Tag color="purple">Market Leaders</Tag>}>
            {chartData && (
              <Column
                data={chartData.makeDistribution}
                xField="make"
                yField="count"
                height={300}
                columnStyle={{
                  fill: '#1890ff',
                  fillOpacity: 0.8,
                }}
                label={{
                  position: 'top',
                  style: {
                    fill: '#000',
                    opacity: 0.6,
                  },
                }}
              />
            )}
          </Card>
        </Col>
        <Col xs={24} lg={8}>
          <Card title="Model Performance" extra={<TrophyOutlined />}>
            <Space direction="vertical" style={{ width: '100%' }}>
              {models.slice(0, 5).map((model, index) => (
                <Card key={model.modelId} size="small">
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div>
                      <Text strong>{model.modelType}</Text>
                      <br />
                      <Text type="secondary" style={{ fontSize: '12px' }}>
                        {new Date(model.createdAt).toLocaleDateString()}
                      </Text>
                    </div>
                    <div style={{ textAlign: 'right' }}>
                      <Progress
                        type="circle"
                        size={40}
                        percent={Math.round((model.metrics.accuracy || 0) * 100)}
                        format={(percent) => `${percent}%`}
                      />
                    </div>
                  </div>
                </Card>
              ))}
              {models.length === 0 && (
                <div style={{ textAlign: 'center', padding: '20px' }}>
                  <ExperimentOutlined style={{ fontSize: '48px', color: '#d9d9d9' }} />
                  <div style={{ marginTop: 16 }}>
                    <Text type="secondary">No models trained yet</Text>
                  </div>
                </div>
              )}
            </Space>
          </Card>
        </Col>
      </Row>

      {/* Quick Actions */}
      <Row style={{ marginTop: 24 }}>
        <Col span={24}>
          <Card title="Quick Actions">
            <Space size="middle">
              <Button type="primary" icon={<ExperimentOutlined />}>
                Train New Model
              </Button>
              <Button icon={<BarChartOutlined />}>
                Explore Data
              </Button>
              <Button icon={<ThunderboltOutlined />}>
                Run Prediction
              </Button>
            </Space>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default Dashboard;