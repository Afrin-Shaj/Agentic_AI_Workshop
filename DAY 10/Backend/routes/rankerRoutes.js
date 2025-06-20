const express = require('express');
const router = express.Router();
const { rankResumes } = require('../controllers/rankerController');
const multer = require('multer');

const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

router.post('/rank-resumes', upload.fields([{ name: 'resumes', maxCount: 10 }]), rankResumes);

module.exports = router;