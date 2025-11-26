import React from 'react';
import { BrowserRouter as Router } from 'react-router-dom';
import AppContent from './AppContent.jsx'; // Import the new AppContent component

function App() {
  return (
    <Router>
      <AppContent />
    </Router>
  );
}

export default App;

