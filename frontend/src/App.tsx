import { Routes, Route } from 'react-router-dom'
import Landing from './pages/Landing'
import Dashboard from './pages/Dashboard'
import Onboarding from './pages/Onboarding'
import Settings from './pages/Settings'

function App() {
  return (
    <>
      {/* Animated star background */}
      <div className="background-container">
        <div className="stars">
          <div className="twinkling"></div>
        </div>
      </div>

      {/* Main app */}
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/onboarding" element={<Onboarding />} />
        <Route path="/settings" element={<Settings />} />
      </Routes>
    </>
  )
}

export default App
