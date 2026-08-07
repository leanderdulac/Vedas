import { useState } from "react";
import { Link, NavLink, Outlet } from "react-router-dom";

const links = [
  { to: "/", label: "Início", end: true },
  { to: "/biblioteca", label: "Biblioteca" },
  { to: "/busca", label: "Busca" },
  { to: "/perguntar", label: "Perguntar" },
  { to: "/sobre", label: "Sobre" },
];

export default function Layout() {
  const [open, setOpen] = useState(false);

  return (
    <div className="app-shell">
      <header className="nav">
        <div className="container nav-inner">
          <Link to="/" className="brand" onClick={() => setOpen(false)}>
            <div className="brand-mark" aria-hidden>
              ॐ
            </div>
            <div className="brand-text">
              <strong>Veda Knowledge</strong>
              <span className="deva">पूर्णमदः पूर्णमिदम्</span>
            </div>
          </Link>

          <button
            type="button"
            className="nav-toggle"
            aria-label="Menu"
            onClick={() => setOpen((v) => !v)}
          >
            ☰
          </button>

          <nav className={`nav-links ${open ? "open" : ""}`}>
            {links.map((l) => (
              <NavLink
                key={l.to}
                to={l.to}
                end={l.end}
                onClick={() => setOpen(false)}
                className={({ isActive }) => (isActive ? "active" : undefined)}
              >
                {l.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>

      <main className="main">
        <Outlet />
      </main>

      <footer className="footer">
        <div className="container footer-inner">
          <div>
            <strong className="display">Veda Knowledge</strong>
            <div className="dim">
              Fontes autorizadas · busca semântica · RAG com citação
            </div>
          </div>
          <div className="dim">
            Use apenas textos com licença válida (public-domain, CC, authorized).
          </div>
        </div>
      </footer>
    </div>
  );
}
