import React, { useState, useEffect } from 'react';
import { useAuth } from './AuthContext.jsx';
import { GoogleLogin } from '@react-oauth/google';

function AuthForm({ onLogin }) {
  const [isLoginView, setIsLoginView] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const { login } = useAuth();

  const handleRegister = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess('');
    try {
      const response = await fetch('http://127.0.0.1:8000/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || 'Registration failed');
      }
      setSuccess('Registration successful! Please log in.');
      setIsLoginView(true); // Switch to login view after successful registration
    } catch (err)
 {
      setError(err.message);
    }
  };

  const handleLogin = async (e) => {
    e.preventDefault();
    console.log("Attempting login...");
    console.log("Email:", email);
    console.log("Password:", password);
    setError('');
    setSuccess('');
    const formData = new URLSearchParams();
    formData.append('username', email);
    formData.append('password', password);
    try {
      const response = await fetch('http://127.0.0.1:8000/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: formData,
      });
      console.log("Raw login response:", response);
      console.log("Login response status:", response.status);
      console.log("Login response status text:", response.statusText);
      const data = await response.json();
      console.log("Login response data:", data);
      if (!response.ok) {
        console.error("Backend login error response:", data);
        throw new Error(data.detail || 'Login failed');
      }
      // Use the login function from AuthContext
      login(data.access_token);
      setSuccess('Login successful! Welcome back.');
      setEmail(''); // Clear fields only on success
      setPassword(''); // Clear fields only on success
    } catch (err) {
      setError(err.message);
    }
  };

  const handleGoogleLoginSuccess = async (credentialResponse) => {
    setError('');
    setSuccess('');
    try {
      // Send the Google token to our backend
      const response = await fetch('http://127.0.0.1:8000/login/google', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ token: credentialResponse.credential }),
      });
      const data = await response.json();
      if (!response.ok) {
          throw new Error(data.detail || 'Google login failed on the backend');
      }
      // Use the token from our backend to log in to the frontend app
      login(data.access_token);
      setSuccess('Google login successful! Welcome.');
    } catch (err) {
      setError(err.message);
    }
  };

  const handleGoogleLoginError = () => {
    setError('Google login failed. Please try again.');
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-900 font-sans">
      <div className="w-full max-w-md p-8 space-y-6 bg-gray-800 rounded-lg shadow-xl">
        <h2 className="text-3xl font-bold text-center text-white">
          {isLoginView ? 'Welcome Back' : 'Create an Account'}
        </h2>

        {error && <p className="text-red-400 text-center bg-red-900/50 p-3 rounded-md border border-red-500/50">{error}</p>}
        {success && <p className="text-green-400 text-center bg-green-900/50 p-3 rounded-md border border-green-500/50">{success}</p>}

        <div className="flex justify-center">
          <GoogleLogin
            onSuccess={handleGoogleLoginSuccess}
            onError={handleGoogleLoginError}
            theme="filled_black"
            text={isLoginView ? "signin_with" : "signup_with"}
            shape="rectangular"
          />
        </div>

        <div className="relative flex py-2 items-center">
            <div className="flex-grow border-t border-gray-600"></div>
            <span className="flex-shrink mx-4 text-gray-400 uppercase text-xs">Or</span>
            <div className="flex-grow border-t border-gray-600"></div>
        </div>

        <form onSubmit={isLoginView ? handleLogin : handleRegister} className="space-y-6">
          <div>
            <label htmlFor="email" className="text-sm font-medium text-gray-300 sr-only">Email address</label>
            <input id="email" name="email" type="email" autoComplete="email" required value={email} onChange={(e) => setEmail(e.target.value)} className="w-full px-3 py-2 mt-1 bg-gray-700 border border-gray-600 rounded-md text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500" placeholder="Email address" />
          </div>
          <div>
            <label htmlFor="password" className="text-sm font-medium text-gray-300 sr-only">Password</label>
            <input id="password" name="password" type="password" autoComplete="current-password" required value={password} onChange={(e) => setPassword(e.target.value)} className="w-full px-3 py-2 mt-1 bg-gray-700 border border-gray-600 rounded-md text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500" placeholder="Password" />
          </div>
          <div>
            <button type="submit" className="w-full py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-cyan-600 hover:bg-cyan-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-800 focus:ring-cyan-500 transition-colors duration-200">
              {isLoginView ? 'Sign In' : 'Create Account'}
            </button>
          </div>
        </form>

        <div className="text-center">
          <button onClick={() => { setIsLoginView(!isLoginView); setError(''); setSuccess(''); }} className="text-sm text-cyan-400 hover:underline">
            {isLoginView ? "Don't have an account? Sign up" : 'Already have an account? Sign In'}
          </button>
        </div>

      </div>
    </div>
  );
}

export default AuthForm; 