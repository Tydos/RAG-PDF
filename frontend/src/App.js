import { useEffect, useState } from 'react';
import Navbar from './components/Navbar';
import UploadSection from './components/UploadSection';
import DocumentsSection from './components/DocumentsSection';
import ChatSection from './components/ChatSection';
import EvalDashboard from './components/EvalDashboard';
import { getHistory } from './api/api';

export default function App() {
  const [refreshKey, setRefreshKey] = useState(0);
  const [view, setView] = useState('chat');
  const [messages, setMessages] = useState([]);

  useEffect(() => {
    getHistory()
      .then(setMessages)
      .catch(() => {});
  }, []);

  return (
    <div className="app-root">
      <Navbar view={view} onViewChange={setView} />
      <div className="app-body">
        {view === 'chat' ? (
          <div className="app-layout">
            <aside className="sidebar">
              <UploadSection onSuccess={() => setRefreshKey((k) => k + 1)} />
              <DocumentsSection refreshKey={refreshKey} />
            </aside>
            <main className="chat-panel">
              <ChatSection messages={messages} setMessages={setMessages} />
            </main>
          </div>
        ) : (
          <EvalDashboard messages={messages} />
        )}
      </div>
    </div>
  );
}
