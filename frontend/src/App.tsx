import { Link, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import HomePage from "./pages/HomePage";
import LibraryPage from "./pages/LibraryPage";
import SearchPage from "./pages/SearchPage";
import AskPage from "./pages/AskPage";
import DocumentPage from "./pages/DocumentPage";
import AboutPage from "./pages/AboutPage";

function NotFound() {
  return (
    <div className="container" style={{ padding: "3rem 1rem", textAlign: "center" }}>
      <h1>Página não encontrada</h1>
      <p className="muted">O caminho solicitado não existe.</p>
      <Link to="/" className="btn btn-primary">
        Voltar ao início
      </Link>
    </div>
  );
}

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<HomePage />} />
        <Route path="biblioteca" element={<LibraryPage />} />
        <Route path="busca" element={<SearchPage />} />
        <Route path="perguntar" element={<AskPage />} />
        <Route path="documento/:id" element={<DocumentPage />} />
        <Route path="sobre" element={<AboutPage />} />
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  );
}
