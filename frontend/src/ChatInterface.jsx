import React, { useState, useEffect, useRef } from "react";
import { useAuth } from "./AuthContext";
import ChatMessage from "./ChatMessage";
import AnimeRecommendationCard from "./AnimeRecommendationCard";

const ChatInterface = ({ chatId }) => {
  const { token } = useAuth();
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [isSending, setIsSending] = useState(false);
  const [recommendations, setRecommendations] = useState(null);
  const messagesEndRef = useRef(null);

  // Load chat messages from backend when chatId changes
  useEffect(() => {
    if (chatId) {
      loadChatMessages();
    }
  }, [chatId]);

  const loadChatMessages = async () => {
    setIsLoading(true);
    try {
      const response = await fetch(
        `http://127.0.0.1:8000/conversations/${chatId}/messages`,
        {
          headers: { Authorization: `Bearer ${token}` },
        }
      );

      if (response.ok) {
        const data = await response.json();
        const formattedMessages = data.map((msg) => ({
          sender: msg.speaker === "user" ? "user" : "bot",
          text: msg.message_text,
        }));
        setMessages(formattedMessages);
      } else {
        // If no messages, show initial greeting
        setMessages([
          {
            sender: "bot",
            text: "Hello! I'm your AI Anime Assistant. How can I help you today?",
          },
        ]);
      }
    } catch (error) {
      console.error("Error loading chat messages:", error);
      setMessages([
        {
          sender: "bot",
          text: "Hello! I'm your AI Anime Assistant. How can I help you today?",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, recommendations]);

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim() || isSending) return;

    const userMessage = { sender: "user", text: input };
    setMessages((prev) => [...prev, userMessage]);
    const originalInput = input;
    setInput("");
    setIsSending(true);
    setRecommendations(null);

    try {
      const response = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          message: originalInput,
          conversation_id: chatId,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to get response");
      }

      const data = await response.json();
      const botMessage = {
        sender: "bot",
        text: data.response,
        action_type: data.action_type,
      };
      setMessages((prev) => [...prev, botMessage]);

      if (data.recommendations?.length > 0) {
        setRecommendations(data.recommendations);
      }
    } catch (error) {
      console.error("Chat error:", error);
      const errorMessage = {
        sender: "bot",
        text: `Sorry, something went wrong: ${error.message}`,
        type: "error",
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsSending(false);
    }
  };

  const handleAnimeAction = async (animeId, action) => {
    try {
      const response = await fetch(
        `http://127.0.0.1:8000/mal/anime/${animeId}/status?status=${action}`,
        {
          method: "PUT",
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );

      if (response.ok) {
        const successMsg = {
          sender: "bot",
          text: `✅ Added to your ${
            action === "watching" ? "Watching" : "Plan to Watch"
          } list!`,
        };
        setMessages((prev) => [...prev, successMsg]);
      }
    } catch (error) {
      console.error("Action failed:", error);
    }
  };

  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center h-screen bg-gray-900 text-white">
        <div className="text-4xl mb-4">💬</div>
        <p className="text-lg">Loading your chat...</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-screen bg-gray-900 text-white">
      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {messages.map((msg, idx) => (
          <ChatMessage
            key={idx}
            sender={msg.sender}
            text={msg.text}
            type={msg.type}
          />
        ))}
        {isSending && (
          <ChatMessage sender="bot" text="⏳ Typing..." type="loading" />
        )}

        {recommendations && recommendations.length > 0 && (
          <div className="mt-8">
            <p className="text-lg font-bold text-indigo-400 mb-4">
              📺 Recommended Anime:
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {recommendations.map((anime) => (
                <AnimeRecommendationCard
                  key={anime.anime_id}
                  anime={anime}
                  onAction={handleAnimeAction}
                />
              ))}
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Form */}
      <form onSubmit={handleSend} className="p-4 bg-gray-800 border-t border-gray-700">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Tell me about your anime..."
            className="flex-1 px-4 py-3 bg-gray-700 text-white rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
            disabled={isSending}
          />
          <button
            type="submit"
            disabled={isSending || !input.trim()}
            className="px-6 py-3 bg-indigo-600 rounded-lg hover:bg-indigo-700 disabled:bg-gray-600 disabled:cursor-not-allowed transition font-semibold"
          >
            Send
          </button>
        </div>
      </form>
    </div>
  );
};

export default ChatInterface;
