import React, { useState, useEffect } from 'react';
import axios from 'axios';
import * as tf from '@tensorflow/tfjs';
import { Container, Row, Col, Form, Button, Card, Table, Spinner, Alert, Badge } from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';

function App() {
  // State for data
  const [irisData, setIrisData] = useState(null);
  const [dataLoaded, setDataLoaded] = useState(false);
  
  // State for model and training
  const [model, setModel] = useState(null);
  const [trainedModelAvailable, setTrainedModelAvailable] = useState(false);
  const [predictions, setPredictions] = useState(null);
  const [customPrediction, setCustomPrediction] = useState(null);
  const [trainingMetrics, setTrainingMetrics] = useState(null);

  // State for UI
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // State for form inputs
  const [formData, setFormData] = useState({
    sepalLength: 5.1,
    sepalWidth: 3.5,
    petalLength: 1.4,
    petalWidth: 0.2,
    epochs: 100,
    learningRate: 0.06,
    firstLayerUnits: 8,
    secondLayerUnits: 10
  });

  // Load iris data on component mount
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const response = await axios.get('/api/iris-data');
        setIrisData(response.data);
        setDataLoaded(true);
        setLoading(false);
      } catch (err) {
        console.error('Error fetching iris data:', err);
        setError('Failed to fetch iris data. Please refresh the page.');
        setLoading(false);
      }
    };
    
    fetchData();
  }, []);

  // Handle form input changes
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value
    });
  };

  // Train model function
  const trainModel = async () => {
    if (!irisData) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const startTime = Date.now();
      
      // Convert data to tensors
      const trainingData = tf.tensor2d(irisData.trainingData.map(item => [
        item.sepal_length, item.sepal_width, item.petal_length, item.petal_width
      ]));
      
      const outputData = tf.tensor2d(irisData.trainingData.map(item => [
        item.species === "setosa" ? 1 : 0,
        item.species === "virginica" ? 1 : 0,
        item.species === "versicolor" ? 1 : 0
      ]));
      
      // Build neural network using a sequential model
      const newModel = tf.sequential();
      
      // Add the first layer with relu activation
      newModel.add(tf.layers.dense({
        inputShape: [4], // Four input features
        units: parseInt(formData.firstLayerUnits),
        activation: 'relu',
      }));
      
      // Add second hidden layer
      newModel.add(tf.layers.dense({
        units: parseInt(formData.secondLayerUnits),
        activation: 'relu',
      }));
      
      // Add the output layer
      newModel.add(tf.layers.dense({
        units: 3, // Three output units for classes
        activation: 'softmax',
      }));
      
      // Compile the model
      newModel.compile({
        optimizer: tf.train.adam(parseFloat(formData.learningRate)),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
      });
      
      // Array to store final metrics
      let finalLoss, finalAccuracy;
      
      // Train the model
      await newModel.fit(trainingData, outputData, {
        epochs: parseInt(formData.epochs),
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            // Store final values
            finalLoss = logs.loss;
            finalAccuracy = logs.acc;
          }
        }
      });
      
      // Calculate training time
      const elapsedTime = Date.now() - startTime;
      
      // Set training metrics
      setTrainingMetrics({
        elapsedTime,
        finalLoss,
        finalAccuracy,
        epochs: parseInt(formData.epochs),
        learningRate: parseFloat(formData.learningRate),
        firstLayerUnits: parseInt(formData.firstLayerUnits),
        secondLayerUnits: parseInt(formData.secondLayerUnits)
      });
      
      // Save the model
      setModel(newModel);
      setTrainedModelAvailable(true);
      
      // Make test predictions
      await makePredictions(newModel);
      
      // Make custom prediction
      await predictCustomInput(newModel);
      
    } catch (err) {
      console.error('Error training model:', err);
      setError('Error during model training: ' + err.message);
    } finally {
      setLoading(false);
    }
  };
  
  // Make predictions on test data
  const makePredictions = async (modelToUse) => {
    if (!modelToUse || !irisData) return;
    
    try {
      const testingData = tf.tensor2d(irisData.testingData.map(item => [
        item.sepal_length, item.sepal_width, item.petal_length, item.petal_width
      ]));
      
      const results = modelToUse.predict(testingData);
      const predictionArray = await results.array();
      
      // Process predictions
      const predictedSpecies = predictionArray.map(row => {
        const highestProbIndex = row.findIndex(val => val === Math.max(...row));
        switch(highestProbIndex) {
          case 0: return 'setosa';
          case 1: return 'virginica';
          case 2: return 'versicolor';
          default: return 'Unknown';
        }
      });
      
      setPredictions({
        species: predictedSpecies.slice(0, 3),
        probabilities: predictionArray.slice(0, 3)
      });
      
    } catch (err) {
      console.error('Error making predictions:', err);
    }
  };
  
  // Predict custom input
  const predictCustomInput = async (modelToUse) => {
    if (!modelToUse) return;
    
    try {
      const customInput = tf.tensor2d([[
        parseFloat(formData.sepalLength),
        parseFloat(formData.sepalWidth),
        parseFloat(formData.petalLength),
        parseFloat(formData.petalWidth)
      ]]);
      
      const prediction = modelToUse.predict(customInput);
      const probabilities = await prediction.array();
      
      const row = probabilities[0];
      const highestProbIndex = row.findIndex(val => val === Math.max(...row));
      
      let species;
      switch(highestProbIndex) {
        case 0: species = 'setosa'; break;
        case 1: species = 'virginica'; break;
        case 2: species = 'versicolor'; break;
        default: species = 'Unknown';
      }
      
      setCustomPrediction({
        probabilities: row,
        predictedSpecies: species
      });
      
    } catch (err) {
      console.error('Error predicting custom input:', err);
    }
  };
  
  // Format probability as percentage
  const formatProbability = (value) => {
    return (value * 100).toFixed(2) + '%';
  };

  return (
    <Container fluid className="px-4 py-5 bg-light">
      <div className="mb-5 text-center">
        <h1 className="fw-bold text-primary">Iris Flower Classification</h1>
      </div>
      
      <Row className="g-4 mb-4">
        <Col md={6}>
          <Card className="shadow-sm border-0 h-100">
            <Card.Header className="bg-primary text-white">
              <h5 className="mb-0">Model Configuration</h5>
            </Card.Header>
            <Card.Body>
              <Form onSubmit={(e) => { e.preventDefault(); trainModel(); }}>
                <Row>
                  <Col md={6}>
                    <Form.Group className="mb-3">
                      <Form.Label className="fw-bold">Epochs</Form.Label>
                      <Form.Control 
                        type="number" 
                        name="epochs" 
                        value={formData.epochs} 
                        onChange={handleInputChange}
                        min="1"
                        max="1000"
                        className="rounded-pill"
                      />
                    </Form.Group>
                  </Col>
                  <Col md={6}>
                    <Form.Group className="mb-3">
                      <Form.Label className="fw-bold">Learning Rate</Form.Label>
                      <Form.Control 
                        type="number" 
                        name="learningRate" 
                        value={formData.learningRate} 
                        onChange={handleInputChange}
                        step="0.01"
                        min="0.01"
                        max="1"
                        className="rounded-pill"
                      />
                    </Form.Group>
                  </Col>
                </Row>
                
                <Row>
                  <Col md={6}>
                    <Form.Group className="mb-3">
                      <Form.Label className="fw-bold">First Layer Units</Form.Label>
                      <Form.Control 
                        type="number" 
                        name="firstLayerUnits" 
                        value={formData.firstLayerUnits} 
                        onChange={handleInputChange}
                        min="1"
                        className="rounded-pill"
                      />
                    </Form.Group>
                  </Col>
                  <Col md={6}>
                    <Form.Group className="mb-3">
                      <Form.Label className="fw-bold">Second Layer Units</Form.Label>
                      <Form.Control 
                        type="number" 
                        name="secondLayerUnits" 
                        value={formData.secondLayerUnits} 
                        onChange={handleInputChange}
                        min="1"
                        className="rounded-pill"
                      />
                    </Form.Group>
                  </Col>
                </Row>
              
                <div className="border-top pt-3 mb-3">
                  <h6 className="text-muted mb-3">Custom Input Data</h6>
                  <Row>
                    <Col md={6}>
                      <Form.Group className="mb-3">
                        <Form.Label>Sepal Length</Form.Label>
                        <Form.Control 
                          type="number" 
                          name="sepalLength" 
                          value={formData.sepalLength} 
                          onChange={handleInputChange}
                          step="0.1"
                          className="rounded-pill"
                        />
                      </Form.Group>
                    </Col>
                    <Col md={6}>
                      <Form.Group className="mb-3">
                        <Form.Label>Sepal Width</Form.Label>
                        <Form.Control 
                          type="number" 
                          name="sepalWidth" 
                          value={formData.sepalWidth} 
                          onChange={handleInputChange}
                          step="0.1"
                          className="rounded-pill"
                        />
                      </Form.Group>
                    </Col>
                  </Row>
                  <Row>
                    <Col md={6}>
                      <Form.Group className="mb-3">
                        <Form.Label>Petal Length</Form.Label>
                        <Form.Control 
                          type="number" 
                          name="petalLength" 
                          value={formData.petalLength} 
                          onChange={handleInputChange}
                          step="0.1"
                          className="rounded-pill"
                        />
                      </Form.Group>
                    </Col>
                    <Col md={6}>
                      <Form.Group className="mb-3">
                        <Form.Label>Petal Width</Form.Label>
                        <Form.Control 
                          type="number" 
                          name="petalWidth" 
                          value={formData.petalWidth} 
                          onChange={handleInputChange}
                          step="0.1"
                          className="rounded-pill"
                        />
                      </Form.Group>
                    </Col>
                  </Row>
                </div>
                
                <div className="d-grid">
                  <Button 
                    variant="primary" 
                    type="submit" 
                    disabled={loading || !dataLoaded}
                    className="rounded-pill py-2 fw-bold"
                  >
                    {loading ? (
                      <>
                        <Spinner as="span" animation="border" size="sm" role="status" aria-hidden="true" />
                        {' '}Training Model...
                      </>
                    ) : 'Train Model and Generate Predictions'}
                  </Button>
                </div>
              </Form>
            </Card.Body>
          </Card>
        </Col>
        
        <Col md={6}>
          <Card className="shadow-sm border-0 h-100">
            <Card.Header className="bg-primary text-white">
              <h5 className="mb-0">Species Reference</h5>
            </Card.Header>
            <Card.Body className="p-0">
              <Table hover className="mb-0">
                <thead className="bg-light">
                  <tr>
                    <th>Species</th>
                    <th>Encoding</th>
                    <th>Characteristics</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td><Badge bg="primary">Setosa</Badge></td>
                    <td><code>[1, 0, 0]</code></td>
                    <td>Short petals, wide sepals</td>
                  </tr>
                  <tr>
                    <td><Badge bg="success">Virginica</Badge></td>
                    <td><code>[0, 1, 0]</code></td>
                    <td>Long petals, narrow sepals</td>
                  </tr>
                  <tr>
                    <td><Badge bg="warning">Versicolor</Badge></td>
                    <td><code>[0, 0, 1]</code></td>
                    <td>Medium petals and sepals</td>
                  </tr>
                </tbody>
              </Table>
            </Card.Body>
          </Card>
        </Col>
      </Row>
      
      <Row>
        <Col>
          {error && (
            <Alert variant="danger" className="mb-4">
              {error}
            </Alert>
          )}
          
          {loading && (
            <div className="text-center my-5 py-5">
              <Spinner animation="border" variant="primary" size="lg" />
              <p className="lead mt-3 text-primary">Training the neural network...</p>
              <p className="text-muted">This may take a few moments depending on the number of epochs</p>
            </div>
          )}
          
          {trainedModelAvailable && !loading && (
            <>
              <Card className="shadow-sm border-0 mb-4">
                <Card.Header className="bg-success text-white d-flex justify-content-between align-items-center">
                  <h5 className="mb-0">Model Performance</h5>
                  <Badge bg="light" text="dark" pill>Training Complete</Badge>
                </Card.Header>
                <Card.Body>
                  <Row className="mb-3">
                    <Col sm={3}>
                      <div className="border rounded p-3 text-center h-100">
                        <div className="text-muted small">Final Accuracy</div>
                        <div className="display-6 fw-bold text-success">
                          {trainingMetrics.finalAccuracy ? 
                            (trainingMetrics.finalAccuracy * 100).toFixed(1) + '%' : 'N/A'}
                        </div>
                      </div>
                    </Col>
                    <Col sm={3}>
                      <div className="border rounded p-3 text-center h-100">
                        <div className="text-muted small">Final Loss</div>
                        <div className="display-6 fw-bold text-danger">
                          {trainingMetrics.finalLoss?.toFixed(4)}
                        </div>
                      </div>
                    </Col>
                    <Col sm={3}>
                      <div className="border rounded p-3 text-center h-100">
                        <div className="text-muted small">Training Time</div>
                        <div className="display-6 fw-bold text-primary">
                          {(trainingMetrics.elapsedTime / 1000).toFixed(1)}s
                        </div>
                      </div>
                    </Col>
                    <Col sm={3}>
                      <div className="border rounded p-3 text-center h-100">
                        <div className="text-muted small">Epochs</div>
                        <div className="display-6 fw-bold text-primary">
                          {trainingMetrics.epochs}
                        </div>
                      </div>
                    </Col>
                  </Row>
                </Card.Body>
              </Card>
              
              {customPrediction && (
                <Card className="shadow-sm border-0 mb-4">
                  <Card.Header className={`bg-${
                    customPrediction.predictedSpecies === 'setosa' ? 'primary' :
                    customPrediction.predictedSpecies === 'virginica' ? 'success' : 'warning'
                  } text-white`}>
                    <h5 className="mb-0">Custom Input Prediction</h5>
                  </Card.Header>
                  <Card.Body>
                    <div className="mb-4 text-center">
                      <h2 className="display-6 mb-3">Predicted Species</h2>
                      <Badge 
                        bg={
                          customPrediction.predictedSpecies === 'setosa' ? 'primary' :
                          customPrediction.predictedSpecies === 'virginica' ? 'success' : 'warning'
                        } 
                        className="px-4 py-2 fs-5"
                      >
                        {customPrediction.predictedSpecies}
                      </Badge>
                    </div>
                    
                    <Row>
                      <Col md={6}>
                        <h6 className="text-muted mb-2">Input Values</h6>
                        <div className="border rounded p-3 mb-3">
                          <Row className="g-2">
                            <Col xs={6}>
                              <div className="small text-muted">Sepal Length</div>
                              <div className="fw-bold">{formData.sepalLength}</div>
                            </Col>
                            <Col xs={6}>
                              <div className="small text-muted">Sepal Width</div>
                              <div className="fw-bold">{formData.sepalWidth}</div>
                            </Col>
                            <Col xs={6}>
                              <div className="small text-muted">Petal Length</div>
                              <div className="fw-bold">{formData.petalLength}</div>
                            </Col>
                            <Col xs={6}>
                              <div className="small text-muted">Petal Width</div>
                              <div className="fw-bold">{formData.petalWidth}</div>
                            </Col>
                          </Row>
                        </div>
                      </Col>
                      <Col md={6}>
                        <h6 className="text-muted mb-2">Probability Distribution</h6>
                        <div className="border rounded p-3">
                          <div className="mb-2">
                            <div className="d-flex justify-content-between mb-1">
                              <small>Setosa</small>
                              <small>{formatProbability(customPrediction.probabilities[0])}</small>
                            </div>
                            <div className="progress" style={{ height: "8px" }}>
                              <div 
                                className="progress-bar bg-primary" 
                                style={{ width: `${customPrediction.probabilities[0] * 100}%` }}
                              ></div>
                            </div>
                          </div>
                          <div className="mb-2">
                            <div className="d-flex justify-content-between mb-1">
                              <small>Virginica</small>
                              <small>{formatProbability(customPrediction.probabilities[1])}</small>
                            </div>
                            <div className="progress" style={{ height: "8px" }}>
                              <div 
                                className="progress-bar bg-success" 
                                style={{ width: `${customPrediction.probabilities[1] * 100}%` }}
                              ></div>
                            </div>
                          </div>
                          <div>
                            <div className="d-flex justify-content-between mb-1">
                              <small>Versicolor</small>
                              <small>{formatProbability(customPrediction.probabilities[2])}</small>
                            </div>
                            <div className="progress" style={{ height: "8px" }}>
                              <div 
                                className="progress-bar bg-warning" 
                                style={{ width: `${customPrediction.probabilities[2] * 100}%` }}
                              ></div>
                            </div>
                          </div>
                        </div>
                      </Col>
                    </Row>
                  </Card.Body>
                </Card>
              )}
              
              {predictions && (
                <Card className="shadow-sm border-0">
                  <Card.Header className="bg-dark text-white">
                    <h5 className="mb-0">Iris-Testing Data Predictions</h5>
                  </Card.Header>
                  <Card.Body className="p-0">
                    <Table hover responsive className="mb-0">
                      <thead className="bg-light">
                        <tr>
                          <th>#</th>
                          <th>Predicted Species</th>
                          <th>Probabilities</th>
                        </tr>
                      </thead>
                      <tbody>
                        {predictions.species.map((species, index) => (
                          <tr key={index}>
                            <td className="fw-bold">{index + 1}</td>
                            <td>
                              <Badge 
                                bg={
                                  species === 'setosa' ? 'primary' :
                                  species === 'virginica' ? 'success' : 'warning'
                                }
                                pill
                              >
                                {species}
                              </Badge>
                            </td>
                            <td>
                              <div className="d-flex flex-column gap-1">
                                <div className="d-flex align-items-center">
                                  <div className="w-25 text-end pe-2 small">Setosa:</div>
                                  <div className="progress flex-grow-1" style={{ height: "6px" }}>
                                    <div 
                                      className="progress-bar bg-primary" 
                                      style={{ width: `${predictions.probabilities[index][0] * 100}%` }}
                                    ></div>
                                  </div>
                                  <div className="w-25 ps-2 small">{formatProbability(predictions.probabilities[index][0])}</div>
                                </div>
                                <div className="d-flex align-items-center">
                                  <div className="w-25 text-end pe-2 small">Virginica:</div>
                                  <div className="progress flex-grow-1" style={{ height: "6px" }}>
                                    <div 
                                      className="progress-bar bg-success" 
                                      style={{ width: `${predictions.probabilities[index][1] * 100}%` }}
                                    ></div>
                                  </div>
                                  <div className="w-25 ps-2 small">{formatProbability(predictions.probabilities[index][1])}</div>
                                </div>
                                <div className="d-flex align-items-center">
                                  <div className="w-25 text-end pe-2 small">Versicolor:</div>
                                  <div className="progress flex-grow-1" style={{ height: "6px" }}>
                                    <div 
                                      className="progress-bar bg-warning" 
                                      style={{ width: `${predictions.probabilities[index][2] * 100}%` }}
                                    ></div>
                                  </div>
                                  <div className="w-25 ps-2 small">{formatProbability(predictions.probabilities[index][2])}</div>
                                </div>
                              </div>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </Table>
                  </Card.Body>
                </Card>
              )}
            </>
          )}
        </Col>
      </Row>
      
      <footer className="mt-5 pt-3 border-top text-center text-muted">
      <p>COMP308 Lab4 Winter 2025 © | Tanya Truong - 301298361</p>
      </footer>
    </Container>
  );
}

export default App;