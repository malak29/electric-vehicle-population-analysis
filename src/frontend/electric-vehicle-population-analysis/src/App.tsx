import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Provider } from 'react-redux';
import { ConfigProvider, Layout, Menu, theme } from 'antd';
import {
  DashboardOutlined,
  BarChartOutlined,
  ExperimentOutlined,
  SettingOutlined,
  DatabaseOutlined
} from '@ant-design/icons';

import { store } from './store/store';
import Dashboard from './pages/Dashboard';
import DataExploration from './pages/DataExploration';
import ModelTraining from './pages/ModelTraining';
import ModelResults from './pages/ModelResults';
import DataManagement from './pages/DataManagement';
import './App.css';

const { Header, Sider, Content } = Layout;

const App: React.FC = () => {
  const [collapsed, setCollapsed] = React.useState(false);

  const menuItems = [
    {
      key: '/',
      icon: <DashboardOutlined />,
      label: 'Dashboard',
    },
    {
      key: '/data-exploration',
      icon: <BarChartOutlined />,
      label: 'Data Exploration',
    },
    {
      key: '/model-training',
      icon: <ExperimentOutlined />,
      label: 'Model Training',
    },
    {
      key: '/model-results',
      icon: <SettingOutlined />,
      label: 'Model Results',
    },
    {
      key: '/data-management',
      icon: <DatabaseOutlined />,
      label: 'Data Management',
    },
  ];

  return (
    <ConfigProvider
      theme={{
        algorithm: theme.defaultAlgorithm,
        token: {
          colorPrimary: '#1890ff',
          colorSuccess: '#52c41a',
          colorWarning: '#faad14',
          colorError: '#f5222d',
        },
      }}
    >
      <Provider store={store}>
        <Router>
          <Layout style={{ minHeight: '100vh' }}>
            <Sider
              collapsible
              collapsed={collapsed}
              onCollapse={(value) => setCollapsed(value)}
              theme="dark"
            >
              <div
                style={{
                  height: 32,
                  margin: 16,
                  background: 'rgba(255, 255, 255, 0.2)',
                  borderRadius: 6,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'white',
                  fontWeight: 'bold',
                }}
              >
                {collapsed ? 'EV' : 'EV Analysis'}
              </div>
              <Menu
                theme="dark"
                defaultSelectedKeys={['/']}
                mode="inline"
                items={menuItems}
                onClick={({ key }) => {
                  window.location.pathname = key;
                }}
              />
            </Sider>
            <Layout>
              <Header
                style={{
                  padding: 0,
                  background: '#fff',
                  display: 'flex',
                  alignItems: 'center',
                  paddingLeft: 16,
                  boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                }}
              >
                <h1 style={{ margin: 0, color: '#1890ff' }}>
                  Electric Vehicle Population Analysis
                </h1>
              </Header>
              <Content
                style={{
                  margin: '24px 16px',
                  padding: 24,
                  minHeight: 280,
                  background: '#fff',
                  borderRadius: 6,
                }}
              >
                <Routes>
                  <Route path="/" element={<Dashboard />} />
                  <Route path="/data-exploration" element={<DataExploration />} />
                  <Route path="/model-training" element={<ModelTraining />} />
                  <Route path="/model-results" element={<ModelResults />} />
                  <Route path="/data-management" element={<DataManagement />} />
                </Routes>
              </Content>
            </Layout>
          </Layout>
        </Router>
      </Provider>
    </ConfigProvider>
  );
};

export default App;