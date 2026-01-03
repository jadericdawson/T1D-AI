import type { Config } from 'tailwindcss'

const config: Config = {
  darkMode: ["class"],
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    container: {
      center: true,
      padding: "2rem",
      screens: {
        "2xl": "1400px",
      },
    },
    extend: {
      colors: {
        // shadcn/ui base colors
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
        // T1D-AI Glucose Range Colors
        glucose: {
          'critical-low': '#dc2626',    // Red - <54 mg/dL
          'low': '#f97316',             // Orange - 54-70
          'in-range': '#00c6ff',        // Cyan - 70-180
          'high': '#eab308',            // Yellow - 180-250
          'critical-high': '#dc2626',   // Red - >250
        },
        // T1D-AI Metric Colors
        metrics: {
          'iob': '#3b82f6',             // Blue for IOB
          'cob': '#14b8a6',             // Teal for COB
          'isf': '#8b5cf6',             // Purple for ISF
          'insulin': '#f97316',         // Orange for insulin
          'carbs': '#22c55e',           // Green for carbs
          'prediction': '#a855f7',      // Purple for predictions
        },
        // JaderBot cyan accent
        cyan: {
          DEFAULT: '#00c6ff',
          dark: '#0056b3',
        },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      keyframes: {
        "accordion-down": {
          from: { height: "0" },
          to: { height: "var(--radix-accordion-content-height)" },
        },
        "accordion-up": {
          from: { height: "var(--radix-accordion-content-height)" },
          to: { height: "0" },
        },
        "fade-in": {
          from: { opacity: "0", transform: "translateY(20px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
        "glow": {
          "0%, 100%": { boxShadow: "0 0 5px #00c6ff, 0 0 10px #00c6ff" },
          "50%": { boxShadow: "0 0 20px #00c6ff, 0 0 30px #00c6ff" },
        },
        "pulse-ring": {
          "0%": { transform: "scale(0.95)", boxShadow: "0 0 0 0 rgba(0, 198, 255, 0.5)" },
          "70%": { transform: "scale(1)", boxShadow: "0 0 0 10px rgba(0, 198, 255, 0)" },
          "100%": { transform: "scale(0.95)", boxShadow: "0 0 0 0 rgba(0, 198, 255, 0)" },
        },
        "move-twink-back": {
          from: { backgroundPosition: "0 0" },
          to: { backgroundPosition: "-10000px 5000px" },
        },
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
        "fade-in": "fade-in 0.5s ease-out",
        "glow": "glow 2s ease-in-out infinite",
        "pulse-ring": "pulse-ring 2s ease-out infinite",
        "twinkle": "move-twink-back 200s linear infinite",
      },
      fontFamily: {
        'playfair': ['"Playfair Display"', 'serif'],
        'inter': ['Inter', 'sans-serif'],
        'orbitron': ['Orbitron', 'sans-serif'],
      },
      backdropBlur: {
        'glass': '10px',
      },
    },
  },
  plugins: [
    require("tailwindcss-animate"),
  ],
}

export default config
