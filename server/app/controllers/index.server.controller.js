// Load the iris data
const iris = require('../../iris.json');
const irisTesting = require('../../iris-testing.json');
var lossValue;
var elapsedTime;

// Method to serve training and testing data
exports.getIrisData = function(req, res) {
    res.json({
        trainingData: iris,
        testingData: irisTesting
    });
};

// Method to handle prediction parameters
exports.trainAndPredict = function(req, res) {
    // Get parameters from query string or use defaults
    const epochs = parseInt(req.query.epochs) || 100;
    const learningRate = parseFloat(req.query.learningRate) || 0.06;
    const firstLayerUnits = parseInt(req.query.firstLayerUnits) || 8;
    const secondLayerUnits = parseInt(req.query.secondLayerUnits) || 10;
    
    console.log(`Received parameters: epochs=${epochs}, learning rate=${learningRate}, first layer units=${firstLayerUnits}, second layer units=${secondLayerUnits}`);
    
    // Pass the parameters back to the client
    // The actual training will happen on the client side using TensorFlow.js
    
    res.json({
        params: {
            epochs: epochs,
            learningRate: learningRate,
            firstLayerUnits: firstLayerUnits,
            secondLayerUnits: secondLayerUnits
        },
        message: "Parameters received. Training will be performed on the client."
    });
};