import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Activity, Brain, TrendingUp, Shield, Zap, Heart } from 'lucide-react'
import { Button } from '@/components/ui/button'

const fadeInUp = {
  hidden: { opacity: 0, y: 30 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.6 } }
}

const staggerContainer = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.15 } }
}

export default function Landing() {
  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="hero-gradient min-h-screen flex flex-col items-center justify-center px-6 py-20">
        <motion.div
          initial="hidden"
          animate="visible"
          variants={staggerContainer}
          className="text-center max-w-4xl mx-auto"
        >
          {/* Logo/Icon */}
          <motion.div variants={fadeInUp} className="mb-8">
            <div className="w-24 h-24 mx-auto rounded-full bg-gradient-to-r from-cyan to-blue-600 flex items-center justify-center animate-pulse-glow">
              <Activity className="w-12 h-12 text-white" />
            </div>
          </motion.div>

          {/* Title */}
          <motion.h1 variants={fadeInUp} className="hero-title mb-6">
            T1D-AI
          </motion.h1>

          <motion.p variants={fadeInUp} className="hero-subtitle mx-auto mb-4">
            Intelligent Diabetes Management
          </motion.p>

          <motion.p variants={fadeInUp} className="text-lg text-gray-400 max-w-2xl mx-auto mb-12">
            Living with Type 1 Diabetes is a 24/7 job. Every meal, every workout, every moment
            requires decisions. T1D-AI uses machine learning to predict your glucose levels,
            calculate optimal doses, and give you back the mental freedom to live your life.
          </motion.p>

          {/* CTA Buttons */}
          <motion.div variants={fadeInUp} className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link to="/onboarding">
              <Button className="btn-primary text-lg px-8 py-6">
                Get Started Free
              </Button>
            </Link>
            <Link to="/dashboard">
              <Button variant="outline" className="btn-secondary text-lg px-8 py-6">
                View Demo
              </Button>
            </Link>
          </motion.div>
        </motion.div>

        {/* Scroll indicator */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.5 }}
          className="absolute bottom-10"
        >
          <div className="w-6 h-10 border-2 border-gray-500 rounded-full flex justify-center">
            <motion.div
              animate={{ y: [0, 12, 0] }}
              transition={{ duration: 1.5, repeat: Infinity }}
              className="w-1.5 h-3 bg-cyan rounded-full mt-2"
            />
          </div>
        </motion.div>
      </section>

      {/* The T1D Story Section */}
      <section className="py-24 px-6">
        <div className="max-w-6xl mx-auto">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={staggerContainer}
            className="grid md:grid-cols-2 gap-16 items-center"
          >
            <motion.div variants={fadeInUp}>
              <h2 className="font-playfair text-4xl font-bold mb-6 text-white">
                The Reality of <span className="text-cyan">Type 1 Diabetes</span>
              </h2>
              <div className="space-y-4 text-gray-300">
                <p>
                  Type 1 Diabetes isn't a lifestyle disease. It's an autoimmune condition where the
                  body's immune system attacks the insulin-producing cells in the pancreas. Without
                  insulin, glucose can't enter cells, leading to dangerously high blood sugar.
                </p>
                <p>
                  People with T1D must make <strong className="text-cyan">300+ decisions daily</strong> about
                  their diabetes management. How many carbs are in this meal? What's my insulin
                  sensitivity right now? Will this workout spike or drop my blood sugar?
                </p>
                <p>
                  The mental burden is exhausting. <strong className="text-cyan">T1D-AI is here to help.</strong>
                </p>
              </div>
            </motion.div>

            <motion.div variants={fadeInUp} className="glass-card p-8">
              <h3 className="text-2xl font-bold mb-6 text-cyan">Daily Challenges</h3>
              <ul className="space-y-4">
                {[
                  "Predicting glucose response to meals",
                  "Calculating insulin doses accurately",
                  "Managing exercise and activity",
                  "Handling stress and illness",
                  "Preventing dangerous lows overnight",
                  "Avoiding burnout and decision fatigue"
                ].map((challenge, i) => (
                  <li key={i} className="flex items-center gap-3 text-gray-300">
                    <div className="w-2 h-2 rounded-full bg-cyan" />
                    {challenge}
                  </li>
                ))}
              </ul>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-24 px-6 bg-gradient-to-b from-transparent to-slate-900/50">
        <div className="max-w-6xl mx-auto">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={staggerContainer}
            className="text-center mb-16"
          >
            <motion.h2 variants={fadeInUp} className="font-playfair text-4xl font-bold mb-4 text-white">
              AI-Powered <span className="text-cyan">Diabetes Management</span>
            </motion.h2>
            <motion.p variants={fadeInUp} className="text-gray-400 max-w-2xl mx-auto">
              Our machine learning models learn from your unique patterns to provide
              personalized predictions and recommendations.
            </motion.p>
          </motion.div>

          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={staggerContainer}
            className="grid md:grid-cols-3 gap-8"
          >
            {[
              {
                icon: <TrendingUp className="w-8 h-8" />,
                title: "Glucose Predictions",
                description: "LSTM neural networks predict your glucose 5, 10, and 15 minutes ahead with high accuracy."
              },
              {
                icon: <Brain className="w-8 h-8" />,
                title: "ISF Prediction",
                description: "Your Insulin Sensitivity Factor changes throughout the day. Our AI predicts it in real-time."
              },
              {
                icon: <Zap className="w-8 h-8" />,
                title: "Smart Dosing",
                description: "Get correction dose recommendations that account for Insulin on Board and Carbs on Board."
              },
              {
                icon: <Activity className="w-8 h-8" />,
                title: "Real-Time Monitoring",
                description: "Beautiful charts showing your glucose trends, predictions, and treatments in one view."
              },
              {
                icon: <Shield className="w-8 h-8" />,
                title: "Smart Alerts",
                description: "Get notified of predicted highs and lows before they happen, not after."
              },
              {
                icon: <Heart className="w-8 h-8" />,
                title: "AI Insights",
                description: "GPT-powered analysis of your patterns with personalized recommendations."
              }
            ].map((feature, i) => (
              <motion.div
                key={i}
                variants={fadeInUp}
                className="glass-card text-center group"
              >
                <div className="w-16 h-16 mx-auto mb-6 rounded-full bg-gradient-to-r from-cyan/20 to-blue-600/20 flex items-center justify-center text-cyan group-hover:animate-glow transition-all">
                  {feature.icon}
                </div>
                <h3 className="text-xl font-bold mb-3 text-white">{feature.title}</h3>
                <p className="text-gray-400">{feature.description}</p>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="py-24 px-6">
        <div className="max-w-6xl mx-auto">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={staggerContainer}
            className="text-center mb-16"
          >
            <motion.h2 variants={fadeInUp} className="font-playfair text-4xl font-bold mb-4 text-white">
              Get Started in <span className="text-cyan">3 Simple Steps</span>
            </motion.h2>
          </motion.div>

          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={staggerContainer}
            className="grid md:grid-cols-3 gap-8"
          >
            {[
              {
                step: "01",
                title: "Connect Your CGM",
                description: "Link your Gluroo account to sync your glucose data automatically."
              },
              {
                step: "02",
                title: "Set Your Profile",
                description: "Enter your insulin settings, target range, and notification preferences."
              },
              {
                step: "03",
                title: "Let AI Help",
                description: "Watch as our models learn your patterns and provide personalized insights."
              }
            ].map((item, i) => (
              <motion.div
                key={i}
                variants={fadeInUp}
                className="glass-card relative overflow-hidden"
              >
                <span className="absolute -top-4 -left-4 text-8xl font-bold text-cyan/10">
                  {item.step}
                </span>
                <div className="relative z-10">
                  <h3 className="text-xl font-bold mb-3 text-white">{item.title}</h3>
                  <p className="text-gray-400">{item.description}</p>
                </div>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 px-6">
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={fadeInUp}
          className="max-w-4xl mx-auto text-center glass-card py-16"
        >
          <h2 className="font-playfair text-4xl font-bold mb-4 text-white">
            Ready to Take Control?
          </h2>
          <p className="text-gray-400 mb-8 max-w-xl mx-auto">
            Join the community of T1D warriors using AI to manage their diabetes more effectively.
          </p>
          <Link to="/onboarding">
            <Button className="btn-primary text-lg px-10 py-6">
              Start Your Journey
            </Button>
          </Link>
        </motion.div>
      </section>

      {/* Footer */}
      <footer className="py-8 px-6 border-t border-gray-800">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <Activity className="w-6 h-6 text-cyan" />
            <span className="font-bold text-white">T1D-AI</span>
          </div>
          <p className="text-gray-500 text-sm">
            Built with love for the T1D community. Not medical advice.
          </p>
        </div>
      </footer>
    </div>
  )
}
