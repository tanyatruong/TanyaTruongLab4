// express.js -- Express app initialization code
//
// Load the module dependencies
const config = require('./config');
const express = require('express');
const morgan = require('morgan');
const compress = require('compression');
const bodyParser = require('body-parser');
const methodOverride = require('method-override');
const session = require('express-session');
const cors = require('cors');

// Define the Express configuration method
module.exports = function() {
    // Create a new Express application instance
    const app = express();

    // Use CORS middleware
    app.use(cors());

    // Use the 'NODE_ENV' variable to activate the 'morgan' logger or 'compress' middleware
    if (process.env.NODE_ENV === 'development') {
        app.use(morgan('dev'));
    } else if (process.env.NODE_ENV === 'production') {
        app.use(compress());
    }

    // Use the 'body-parser' and 'method-override' middleware functions
    app.use(bodyParser.urlencoded({
        extended: true
    }));
    app.use(bodyParser.json());
    app.use(methodOverride());

    // Configure the 'session' middleware with proper secret
    app.use(session({
        saveUninitialized: true,
        resave: true,
        secret: config.sessionSecret || 'developmentSessionSecret' // Provide a fallback secret
    }));

    // Set the application view engine and 'views' folder
    app.set('views', './app/views');
    app.set('view engine', 'ejs');

    // Load the routing files
    require('../app/routes/index.server.routes.js')(app);

    // Configure static file serving
    app.use(express.static('./public'));

    // Return the Express application instance
    return app;
};