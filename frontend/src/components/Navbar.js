const ChatIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
  </svg>
);

const EvalIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <line x1="18" y1="20" x2="18" y2="10"/>
    <line x1="12" y1="20" x2="12" y2="4"/>
    <line x1="6"  y1="20" x2="6"  y2="14"/>
  </svg>
);


export default function Navbar({ view, onViewChange }) {
  return (
    <nav className="navbar" aria-label="Main navigation">
      <span className="navbar-brand">Antares</span>
      <div className="navbar-links">
        <button
          className={`navbar-link${view === 'chat' ? ' active' : ''}`}
          onClick={() => onViewChange('chat')}
          aria-current={view === 'chat' ? 'page' : undefined}
        >
          <ChatIcon />
          Chat
        </button>
        <button
          className={`navbar-link${view === 'eval' ? ' active' : ''}`}
          onClick={() => onViewChange('eval')}
          aria-current={view === 'eval' ? 'page' : undefined}
        >
          <EvalIcon />
          Evaluation
        </button>
      </div>
    </nav>
  );
}
