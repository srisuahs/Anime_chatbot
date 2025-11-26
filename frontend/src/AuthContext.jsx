import React, { createContext, useState, useContext, useEffect } from 'react';

// 1. Create the context
// This is the object that components will use to get the auth state.
const AuthContext = createContext(null);

// 2. Create the Provider component
// This component will wrap our entire application and manage the auth state.
export function AuthProvider({ children }) {
  // Initialize state from localStorage to keep the user logged in across page refreshes
  const [token, setToken] = useState(localStorage.getItem('authToken'));

  // Use an effect to automatically update localStorage whenever the token changes
  useEffect(() => {
    if (token) {
      // If a token exists, store it.
      localStorage.setItem('authToken', token);
    } else {
      // If the token is null (e.g., after logout), remove it from storage.
      localStorage.removeItem('authToken');
    }
  }, [token]);

  // Function to update the token when a user logs in
  const login = (newToken) => {
    setToken(newToken);
  };

  // Function to clear the token when a user logs out
  const logout = () => {
    setToken(null);
  };

  // The value that will be provided to all consuming components
  const value = { token, login, logout };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

// 3. Create a custom hook for easy access
// This simplifies how other components get the auth state.
// Instead of `useContext(AuthContext)`, they can just call `useAuth()`.
export function useAuth() {
  return useContext(AuthContext);
}

