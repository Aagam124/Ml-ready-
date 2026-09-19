import { createContext, useContext, useEffect, useState } from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { getMe, loginUser, registerUser } from './api';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const token = localStorage.getItem('mlready_token');
    if (!token) {
      setLoading(false);
      return;
    }
    getMe()
      .then((profile) => setUser(profile))
      .catch(() => {
        localStorage.removeItem('mlready_token');
      })
      .finally(() => setLoading(false));
  }, []);

  async function login(username, password) {
    const { access_token } = await loginUser({ username, password });
    localStorage.setItem('mlready_token', access_token);
    const profile = await getMe();
    setUser(profile);
    return profile;
  }

  async function register(username, email, password) {
    await registerUser({ username, email, password });
    return login(username, password);
  }

  function logout() {
    localStorage.removeItem('mlready_token');
    localStorage.removeItem('mlready_last_dataset');
    setUser(null);
  }

  return (
    <AuthContext.Provider value={{ user, loading, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}

export function ProtectedRoute({ children }) {
  const { user, loading } = useAuth();
  const location = useLocation();

  if (loading) return null; // avoid a login-page flash while /auth/me resolves
  if (!user) return <Navigate to="/login" state={{ from: location }} replace />;
  return children;
}
