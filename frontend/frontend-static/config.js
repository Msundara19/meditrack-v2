// Configuration for deployment
const CONFIG = {
    // Local development
    development: {
        API_URL: 'http://localhost:8000'
    },
    // Production (update after deploying backend)
    production: {
        API_URL: 'https://meditrack-backend.onrender.com'
    }
};

// Auto-detect environment
const ENV = window.location.hostname === 'localhost' ? 'development' : 'production';
const API_URL = CONFIG[ENV].API_URL;