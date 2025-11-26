import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate, Link } from 'react-router-dom';
import { useAuth } from './AuthContext.jsx';
import Toast from './Toast.jsx';

const SettingsPage = () => {
  const { token } = useAuth();
  const [malUsername, setMalUsername] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [toast, setToast] = useState({ message: '', type: '', isVisible: false });
  
  const location = useLocation();
  const navigate = useNavigate();

  // Effect to show toast messages passed from other pages (e.g., after MAL connect)
  useEffect(() => {
    if (location.state?.successMessage) {
      setToast({ message: location.state.successMessage, type: 'success', isVisible: true });
      setMalUsername(location.state.malUsername); // Update username from redirect
      navigate(location.pathname, { replace: true, state: {} });
    }
     if (location.state?.errorMessage) {
      setToast({ message: location.state.errorMessage, type: 'error', isVisible: true });
      navigate(location.pathname, { replace: true, state: {} });
    }
  }, [location, navigate]);

  // Effect to fetch the user's current MAL connection status on load
  useEffect(() => {
    const fetchMalStatus = async () => {
      if (!token) return;
      setIsLoading(true);
      try {
        const response = await fetch('http://127.0.0.1:8000/mal/tokens', {
          headers: { 'Authorization': `Bearer ${token}` }
        });
        if (response.ok) {
          const data = await response.json();
          setMalUsername(data.username);
        } else {
          setMalUsername(null);
        }
      } catch (error) {
        console.error("Failed to fetch MAL status:", error);
        setToast({ message: 'Could not fetch MAL status.', type: 'error', isVisible: true });
      } finally {
        setIsLoading(false);
      }
    };
    fetchMalStatus();
  }, [token]);

  // Function to start the MAL connection flow
  const handleConnectMAL = async () => {
    try {
      const state = Array.from(window.crypto.getRandomValues(new Uint8Array(16)), byte => byte.toString(16).padStart(2, '0')).join('');
      const code_challenge = Array.from(window.crypto.getRandomValues(new Uint8Array(48)), byte => byte.toString(16).padStart(2, '0')).join('');

      localStorage.setItem('mal_auth_state', state);
      localStorage.setItem('mal_code_verifier', code_challenge);

      const response = await fetch('http://127.0.0.1:8000/mal/auth-url', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ state, code_challenge }),
      });
      const data = await response.json();
      if (data.authorization_url) {
        window.location.href = data.authorization_url;
      }
    } catch (error) {
      console.error("Failed to get MAL auth URL:", error);
      setToast({ message: 'Failed to start MAL connection.', type: 'error', isVisible: true });
    }
  };
  
  // Function to disconnect from MAL
  const handleDisconnectMAL = async () => {
     try {
      const response = await fetch('http://127.0.0.1:8000/mal/tokens', {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (!response.ok) throw new Error('Failed to disconnect.');
      setMalUsername(null);
      setToast({ message: 'Successfully disconnected from MyAnimeList.', type: 'success', isVisible: true });
    } catch (error) {
      console.error("Failed to disconnect MAL:", error);
      setToast({ message: error.message, type: 'error', isVisible: true });
    }
  };

  return (
    <div className="settings-page p-6 bg-gray-900 min-h-screen text-white">
       {toast.isVisible && <Toast message={toast.message} type={toast.type} onClose={() => setToast({ ...toast, isVisible: false })} />}
      <h2 className="text-3xl font-bold mb-8 text-center">Settings</h2>

      <div className="max-w-2xl mx-auto bg-gray-800 p-8 rounded-lg shadow-lg">
        <h3 className="text-2xl font-semibold mb-4 border-b border-gray-700 pb-2">MyAnimeList Integration</h3>
        {isLoading ? (
          <p className="text-gray-400">Checking connection status...</p>
        ) : malUsername ? (
          <div className="flex flex-col sm:flex-row items-center justify-between">
            <p className="text-lg text-green-400 mb-4 sm:mb-0">
              Connected as: <span className="font-bold">{malUsername}</span>
            </p>
            <button
              onClick={handleDisconnectMAL}
              className="px-6 py-2 font-semibold text-white bg-red-600 rounded-md hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-800 focus:ring-red-500 transition-colors"
            >
              Disconnect
            </button>
          </div>
        ) : (
          <div className="flex flex-col sm:flex-row items-center justify-between">
            <p className="text-lg text-gray-400 mb-4 sm:mb-0">Your account is not connected to MyAnimeList.</p>
            <button
              onClick={handleConnectMAL}
              className="px-6 py-2 font-semibold text-white bg-blue-600 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-800 focus:ring-blue-500 transition-colors"
            >
              Connect Now
            </button>
          </div>
        )}
      </div>
       <div className="text-center mt-8">
            <Link to="/" className="text-blue-400 hover:underline">
                &larr; Back to Chat
            </Link>
        </div>
    </div>
  );
};

export default SettingsPage;

