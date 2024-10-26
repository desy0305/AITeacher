# AI Homework Analysis System

## Overview
The AI Homework Analysis System is an advanced application that combines computer vision, optical character recognition (OCR), and natural language processing (NLP) to analyze and grade homework assignments. The system provides automated assessment capabilities with support for both single submissions and batch processing, specifically optimized for Bulgarian language content.

## Features

### Image Processing & OCR
- Support for multiple image formats (JPEG, PNG, PDF)
- Automatic image enhancement and preprocessing
- Separate processing for printed and handwritten text
- Multi-language OCR with Bulgarian language support
- High-accuracy text extraction
- Intelligent image preprocessing pipeline:
  - Adaptive thresholding
  - Morphological operations
  - Noise reduction
  - Contrast enhancement
  - Separate masks for printed and handwritten content

### Analysis Capabilities
- Comprehensive content analysis
- Subject-specific assessment
- Sentiment analysis
- Detailed feedback generation
- Multiple evaluation criteria:
  - Completeness (30% weight)
  - Understanding (30% weight)
  - Clarity (20% weight)
  - Terminology usage (20% weight)
  - Overall quality
- Automatic translation between Bulgarian and English
- Key points identification
- Areas for improvement detection
- Personalized suggestions generation

### User Interface
- Modern React-based interface
- Intuitive file upload system:
  - Drag-and-drop support
  - File preview
  - Multiple file selection
  - Progress indication
- Support for single and batch processing
- Real-time analysis feedback
- Detailed results visualization:
  - Three-tab interface (Extracted Text, Analysis, Feedback)
  - Progress bars for scoring visualization
  - Organized display of strengths and weaknesses
  - Formatted feedback presentation
- Bulgarian language interface
- Error handling and validation
- Loading states and animations

## Technology Stack

### Backend (app.py)
- FastAPI for API endpoints
- OpenCV for image processing
- Tesseract OCR for printed text
- EasyOCR for handwritten text
- Transformers for NLP tasks:
  - T5 for text analysis
  - BERT for sentiment analysis
- MarianMT for translations
- spaCy for text analysis
- PyTorch for deep learning
- Background task processing
- Comprehensive error handling and logging
- CORS middleware for cross-origin requests

### Frontend (bulgarian-homework-analysis-frontend.tsx)
- React with TypeScript
- Tailwind CSS for styling
- Lucide Icons for UI elements
- Shadcn UI components:
  - Cards
  - Tabs
  - Select
  - Progress
  - Alert
  - Label
  - Button
- Form handling and validation
- Asynchronous API communication
- Error boundary implementation
- Loading state management

## Setup

### Prerequisites
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies
npm install
```

### Configuration
The system uses default configurations for:
- OCR settings
- Analysis parameters
- Grading rubrics:
  - Completeness: 30%
  - Understanding: 30%
  - Clarity: 20%
  - Terminology: 20%
- Language models
- Subject categories:
  - Математика (Mathematics)
  - Литература (Literature)
  - Природни науки (Natural Sciences)
  - История (History)
  - География (Geography)
  - Физика (Physics)
  - Химия (Chemistry)
  - Биология (Biology)

## Usage

### Starting the Application

1. Start the Backend:
```bash
python app.py
```
The backend will start on http://localhost:8000

2. Start the Frontend:
```bash
npm run dev
```

### Using the System

1. Single Submission:
   - Select subject from the dropdown
   - Upload homework image (drag-and-drop or click to select)
   - Click "Анализирай" (Analyze)
   - View analysis results in three tabs:
     - Extracted text (printed question and handwritten answer)
     - Detailed analysis (scores and evaluation points)
     - Generated feedback in Bulgarian

2. Batch Processing:
   - Switch to "Пакетна обработка" (Batch Processing) tab
   - Select multiple files
   - Choose subject
   - Process all submissions simultaneously
   - View batch results
   - Results are automatically saved with timestamps

## API Endpoints

### Single Analysis
```typescript
POST /analyze
Content-Type: multipart/form-data

Parameters:
- file: File
- request: {
  subject?: string
  rubric?: {
    completeness: number  // Default: 0.3
    understanding: number // Default: 0.3
    clarity: number      // Default: 0.2
    terminology: number  // Default: 0.2
  }
}

Response:
{
  success: boolean
  message: string
  data?: {
    extracted_text: {
      printed: string
      handwritten: string
    }
    analysis: {
      completeness: number
      understanding: number
      clarity: number
      terminology: number
      overall_quality: number
      key_points: string[]
      missing_points: string[]
      suggestions: string[]
    }
    feedback_bulgarian: string
  }
}
```

### Batch Analysis
```typescript
POST /analyze-batch
Content-Type: multipart/form-data

Parameters:
- files: File[]
- request: {
  subject?: string
  rubric?: {
    completeness: number
    understanding: number
    clarity: number
    terminology: number
  }
}

Response:
{
  success: boolean
  message: string
  data?: {
    results: Array<{
      filename: string
      analysis: AnalysisResult | null
      error: string | null
    }>
  }
}
```

## Project Structure
```
.
├── app.py                 # Backend implementation
├── requirements.txt       # Python dependencies
├── bulgarian-homework-analysis-frontend.tsx  # Frontend implementation
├── project-readme.md     # Documentation
├── .env                  # Environment variables
├── .gitignore           # Git ignore rules
├── docker-compose.yml    # Docker compose configuration
├── Dockerfile.backend    # Backend Docker configuration
├── Dockerfile.frontend   # Frontend Docker configuration
├── public/              # Public assets
├── src/                 # Frontend source code
│   ├── components/      # React components
│   │   └── ui/         # UI components
│   ├── lib/            # Utility functions
│   └── types/          # TypeScript definitions
└── .dockerignore       # Docker ignore rules
```

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
This project is licensed under the MIT License.
