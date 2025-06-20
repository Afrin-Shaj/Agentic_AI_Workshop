const mongoose = require('mongoose');

const candidateSchema = new mongoose.Schema({
  name: { type: String },
  score: { type: Number },
  matched_skills: [{ type: String }],
  gaps: {
    critical_gaps: [{ type: String }],
    optional_gaps: [{ type: String }],
    experience_gaps: [{ type: String }],
    education_gaps: [{ type: String }],
    recommendations: [{ type: String }],
  },
  resume_data: {
    name: { type: String },
    contact_info: {
      email: { type: String },
      phone: { type: String },
      linkedin: { type: String },
      github: { type: String },
      location: { type: String },
    },
    skills: [{ type: String }],
    experience: [{
      role: { type: String },
      company: { type: String },
      duration: { type: String },
      description: { type: String },
      technologies: [{ type: String }],
    }],
    education: [{
      degree: { type: String },
      institution: { type: String },
      year: { type: String },
      gpa: { type: String },
    }],
    certifications: [{
      name: { type: String },
      issuer: { type: String },
      year: { type: String },
    }],
    projects: [{
      name: { type: String },
      description: { type: String },
      technologies: [{ type: String }],
      outcome: { type: String },
    }],
    languages: [{ type: String }],
    achievements: [{ type: String }],
  },
  score_breakdown: {
    skills: { type: Number },
    experience: { type: Number },
    education: { type: Number },
    projects: { type: Number },
  },
  status: { type: String, enum: ['approved', 'rejected'], required: true },
  rejection_reasons: [{ type: String }]
});

const resumeSchema = new mongoose.Schema({
  job_description: { type: String, required: true },
  candidates: [candidateSchema],
  filter_stats: {
    total_candidates: { type: Number },
    approved: { type: Number },
    rejected: { type: Number },
    approval_rate: { type: Number }
  },
  timestamp: { type: Date, default: Date.now }
});

module.exports = mongoose.model('Resume', resumeSchema);