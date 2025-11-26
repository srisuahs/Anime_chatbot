import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import { AuthProvider } from './AuthContext.jsx'
import { GoogleOAuthProvider } from '@react-oauth/google';

// IMPORTANT: Replace this with the Client ID you got from the Google Cloud Console
const GOOGLE_CLIENT_ID = "73637309130-rktof7ocncis0hqv7ah8oei32ju1al6j.apps.googleusercontent.com";

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <GoogleOAuthProvider clientId={GOOGLE_CLIENT_ID}>
      <AuthProvider>
        <App />
      </AuthProvider>
    </GoogleOAuthProvider>
  </React.StrictMode>,
)

