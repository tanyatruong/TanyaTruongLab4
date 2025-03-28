// Load the 'index' controller
const index = require('../controllers/index.server.controller');

// Define the routes module method
module.exports = function(app) {
    app.get('/', function(req, res) {
        res.render('index', {
            info: "Access the React UI to interact with the neural network"
        });
    });

    app.get('/run', index.trainAndPredict);
    app.get('/iris-data', index.getIrisData);
};