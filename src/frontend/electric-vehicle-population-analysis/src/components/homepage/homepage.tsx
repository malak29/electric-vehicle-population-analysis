import React, { useState } from 'react';
import { handleLoadData, handleCleanData, handleExecuteModel } from '../../utility/modelApi';

const Homepage = () => {
    const [filePath, setFilePath] = useState('');
    const [dfName, setDfName] = useState('');
    const [modelType, setModelType] = useState('');

    return (
        <div>
            <h2>Data Processor</h2>
            <div>
                <input type="text" placeholder="File Path" value={filePath} onChange={e => setFilePath(e.target.value)} />
                <button onClick={handleLoadData}>Load Data</button>
            </div>
            <div>
                <input type="text" placeholder="DataFrame Name" value={dfName} onChange={e => setDfName(e.target.value)} />
                <button onClick={handleCleanData}>Clean Data</button>
            </div>
            <div>
                <select value={modelType} onChange={e => setModelType(e.target.value)}>
                    <option value="">Select Model</option>
                    <option value="decision_tree">Decision Tree</option>
                    <option value="k_means">K-Means</option>
                    <option value="logistic_regression">Logistic Regression</option>
                    <option value="gradient_boosting">Gradient Boosting</option>
                    <option value="linear_regression">Linear Regression</option>
                    <option value="random_forest">Random Forest</option>
                </select>
                <button onClick={handleExecuteModel}>Execute Model</button>
            </div>
        </div>
    );
};

export default Homepage