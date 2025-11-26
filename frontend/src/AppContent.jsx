import React, { useState, useEffect } from "react";
import { Routes, Route, useNavigate, Navigate } from "react-router-dom";
import { useAuth } from "./AuthContext.jsx";
import ChatInterface from "./ChatInterface.jsx";
import SettingsPage from "./SettingsPage.jsx";
import SchedulePage from "./SchedulePage.jsx";
import AuthForm from "./AuthForm.jsx";
import SideBar from "./SideBar.jsx";
import LandingPage from "./LandingPage.jsx";

function AppContent() {
  const { token, login } = useAuth();
  const navigate = useNavigate();
  const [chats, setChats] = useState([]);
  const [activeChatId, setActiveChatId] = useState(null);
  const [isLoadingChats, setIsLoadingChats] = useState(true);
  const [isCollapsed, setIsCollapsed] = useState(false);

  // Load conversations from backend
  useEffect(() => {
    if (token) {
      loadConversations();
    }
  }, [token]);

  const loadConversations = async () => {
    try {
      const response = await fetch("http://127.0.0.1:8000/conversations", {
        headers: { Authorization: `Bearer ${token}` },
      });
      
      if (response.ok) {
        const data = await response.json();
        if (data.length > 0) {
          setChats(data);
          setActiveChatId(data[0].id);
        } else {
          // Create first conversation if none exists
          await handleNewChat();
        }
      }
    } catch (error) {
      console.error("Error loading conversations:", error);
      // Create first conversation on error
      await handleNewChat();
    } finally {
      setIsLoadingChats(false);
    }
  };

  const handleNewChat = async () => {
    try {
      const response = await fetch("http://127.0.0.1:8000/conversations", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ title: "New Chat" }),
      });

      if (response.ok) {
        const newChat = await response.json();
        setChats((prev) => [newChat, ...prev]);
        setActiveChatId(newChat.id);
        navigate("/");
      }
    } catch (error) {
      console.error("Error creating new chat:", error);
    }
  };

  const handleDeleteChat = async (id) => {
    if (chats.length === 1) return alert("You must have at least one chat.");

    try {
      const response = await fetch(`http://127.0.0.1:8000/conversations/${id}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${token}` },
      });

      if (response.ok) {
        const updated = chats.filter((chat) => chat.id !== id);
        setChats(updated);
        if (activeChatId === id) setActiveChatId(updated[0]?.id);
      }
    } catch (error) {
      console.error("Error deleting chat:", error);
    }
  };

  // ✅ Handle MAL redirect
  useEffect(() => {
    const handleMalCallback = async () => {
      const params = new URLSearchParams(window.location.search);
      const code = params.get("code");
      const state = params.get("state");

      if (code && state && token) {
        const verifier = localStorage.getItem("mal_code_verifier");
        const savedState = localStorage.getItem("mal_auth_state");

        if (!verifier || !savedState) {
          window.history.replaceState({}, document.title, "/");
          return;
        }

        if (state !== savedState) {
          window.history.replaceState({}, document.title, "/");
          return;
        }

        try {
          const response = await fetch("http://127.0.0.1:8000/mal/token", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Authorization: `Bearer ${token}`,
            },
            body: JSON.stringify({ code, code_verifier: verifier }),
          });

          if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Failed to exchange MAL token.");
          }

          const data = await response.json();
          navigate("/settings", {
            state: {
              malConnected: true,
              malUsername: data.mal_username,
              successMessage: "MyAnimeList account connected successfully.",
            },
          });

          localStorage.removeItem("mal_code_verifier");
          localStorage.removeItem("mal_auth_state");
          window.history.replaceState({}, document.title, "/");
        } catch (err) {
          navigate("/settings", {
            state: { errorMessage: `Error connecting to MAL: ${err.message}` },
          });
        }
      }
    };

    handleMalCallback();
  }, [token, navigate]);

  // ✅ Public routes (unauthenticated)
  if (!token) {
    return (
      <Routes>
        <Route path="/landing" element={<LandingPage />} />
        <Route path="/login" element={<AuthForm mode="login" onSuccess={login} />} />
        <Route path="/register" element={<AuthForm mode="register" onSuccess={login} />} />
        <Route path="*" element={<Navigate to="/landing" />} />
      </Routes>
    );
  }

  // ✅ Show loading state while fetching conversations
  if (isLoadingChats) {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-900 text-white">
        <div className="text-center">
          <div className="text-4xl mb-4">🎌</div>
          <p className="text-lg">Loading your chats...</p>
        </div>
      </div>
    );
  }

  // ✅ Protected routes with sidebar
  return (
    <div className="flex h-screen bg-gray-900 overflow-hidden">
      <SideBar
        onNewChat={handleNewChat}
        onDeleteChat={handleDeleteChat}
        chats={chats}
        activeChatId={activeChatId}
        setActiveChatId={setActiveChatId}
        isCollapsed={isCollapsed}
        setIsCollapsed={setIsCollapsed}
      />
      <div
        className="flex-1 overflow-auto transition-all duration-300"
        style={{
          marginLeft: isCollapsed ? "4rem" : "0",
        }}
      >
        <Routes>
          <Route path="/" element={<ChatInterface chatId={activeChatId} />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="/schedule" element={<SchedulePage />} />
          <Route path="*" element={<Navigate to="/" />} />
        </Routes>
      </div>
    </div>
  );
}

export default AppContent;
