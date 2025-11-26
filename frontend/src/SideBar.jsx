import React from "react";
import { Link, useLocation } from "react-router-dom";
import {
  MessageSquare,
  Calendar,
  Settings,
  LogOut,
  PlusCircle,
  Trash2,
  Menu,
  X,
} from "lucide-react";
import { useAuth } from "./AuthContext";

const SideBar = ({
  onNewChat,
  onDeleteChat,
  chats,
  activeChatId,
  setActiveChatId,
  isCollapsed,
  setIsCollapsed,
}) => {
  const location = useLocation();
  const { logout } = useAuth();

  const toggleSidebar = () => setIsCollapsed((prev) => !prev);

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffTime = Math.abs(now - date);
    const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));
    
    if (diffDays === 0) return "Today";
    if (diffDays === 1) return "Yesterday";
    if (diffDays < 7) return `${diffDays} days ago`;
    return date.toLocaleDateString();
  };

  return (
    <div
      className={`${
        isCollapsed ? "w-16" : "w-64"
      } bg-gray-800 border-r border-gray-700 flex flex-col transition-all duration-300 fixed h-full z-50 top-0 left-0`}
    >
      {/* Header */}
      <div className="p-4 border-b border-gray-700 flex items-center justify-between">
        {!isCollapsed && (
          <h2 className="text-xl font-bold text-indigo-400">AnimeChat</h2>
        )}
        <button
          onClick={toggleSidebar}
          className="p-2 hover:bg-gray-700 rounded-lg transition ml-auto"
          title={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {isCollapsed ? <Menu size={20} /> : <X size={20} />}
        </button>
      </div>

      {/* New Chat Button */}
      <div className="p-4">
        <button
          onClick={onNewChat}
          className="w-full flex items-center justify-center gap-2 bg-indigo-600 hover:bg-indigo-700 py-2 px-4 rounded-lg transition font-semibold"
          title="Create new chat"
        >
          <PlusCircle size={20} />
          {!isCollapsed && <span>New Chat</span>}
        </button>
      </div>

      {/* Chat List */}
      <div className="flex-1 overflow-y-auto px-2">
        {!isCollapsed && chats.length > 0 && (
          <div className="space-y-1">
            {chats.map((chat) => (
              <div
                key={chat.id}
                className={`${
                  activeChatId === chat.id
                    ? "bg-indigo-600"
                    : "bg-gray-700 hover:bg-gray-600"
                } rounded-lg p-3 cursor-pointer transition group relative`}
                onClick={() => setActiveChatId(chat.id)}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-semibold truncate">
                      {chat.title}
                    </p>
                    <p className="text-xs text-gray-400 mt-1">
                      {formatDate(chat.updated_at)}
                    </p>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeleteChat(chat.id);
                    }}
                    className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-600 rounded transition"
                    title="Delete chat"
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Navigation */}
      <div className="border-t border-gray-700">
        <Link
          to="/"
          className={`flex items-center justify-center gap-3 p-4 hover:bg-gray-700 transition ${
            location.pathname === "/" ? "bg-gray-700" : ""
          }`}
          title="Chat"
        >
          <MessageSquare size={20} />
          {!isCollapsed && <span>Chat</span>}
        </Link>
        <Link
          to="/schedule"
          className={`flex items-center justify-center gap-3 p-4 hover:bg-gray-700 transition ${
            location.pathname === "/schedule" ? "bg-gray-700" : ""
          }`}
          title="Schedule"
        >
          <Calendar size={20} />
          {!isCollapsed && <span>Schedule</span>}
        </Link>
        <Link
          to="/settings"
          className={`flex items-center justify-center gap-3 p-4 hover:bg-gray-700 transition ${
            location.pathname === "/settings" ? "bg-gray-700" : ""
          }`}
          title="Settings"
        >
          <Settings size={20} />
          {!isCollapsed && <span>Settings</span>}
        </Link>
        <button
          onClick={logout}
          className="flex items-center justify-center gap-3 p-4 w-full hover:bg-gray-700 transition text-red-400"
          title="Logout"
        >
          <LogOut size={20} />
          {!isCollapsed && <span>Logout</span>}
        </button>
      </div>
    </div>
  );
};

export default SideBar;
