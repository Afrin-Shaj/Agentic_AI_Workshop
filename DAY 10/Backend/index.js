const express = require('express');
const mongoose = require('mongoose');
const dotenv = require('dotenv');
const aiRoutes = require('./routes/rankerRoutes');
const cors = require('cors');

dotenv.config();

const app = express();

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(cors({
  origin: 'http://localhost:5173',
  methods: ['GET', 'POST'],
  allowedHeaders: ['Content-Type'],
}));

// Connect to MongoDB
mongoose.connect(process.env.MONGODB_URL)
  .then(() => console.log('Connected to MongoDB'))
  .catch((err) => console.error('MongoDB connection error:', err));

// Routes
app.use('/api', aiRoutes);

// Start server
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});