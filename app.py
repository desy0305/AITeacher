import cv2
import numpy as np
import pytesseract
from PIL import Image
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    MarianMTModel,
    MarianTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer
)
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
from datetime import datetime
import asyncio
from dataclasses import dataclass
import easyocr
import spacy
from langdetect import detect
import json
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    success: bool
    message: str
    data: Optional[Dict] = None
    timestamp: datetime = datetime.now()

class BulgarianTextProcessor:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.initialize_models()

    def initialize_models(self):
        try:
            # Translation models
            self.bg_to_en = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-bg-en")
            self.bg_to_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-bg-en")
            self.en_to_bg = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-bg")
            self.en_to_bg_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-bg")
            
            # OCR reader
            self.reader = easyocr.Reader(['bg', 'en'])
            
            # NLP models
            try:
                self.nlp_bg = spacy.load("bg_core_news_lg")
            except:
                self.logger.warning("Bulgarian spaCy model not found. Using English model.")
                self.nlp_bg = spacy.load("en_core_web_sm")
        except Exception as e:
            self.logger.error(f"Model initialization error: {str(e)}")
            raise

    async def translate_to_english(self, text: str) -> str:
        try:
            inputs = self.bg_to_en_tokenizer(text, return_tensors="pt", padding=True)
            outputs = self.bg_to_en.generate(**inputs)
            return self.bg_to_en_tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            self.logger.error(f"Translation error: {str(e)}")
            return text

    async def translate_to_bulgarian(self, text: str) -> str:
        try:
            inputs = self.en_to_bg_tokenizer(text, return_tensors="pt", padding=True)
            outputs = self.en_to_bg.generate(**inputs)
            return self.en_to_bg_tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            self.logger.error(f"Translation error: {str(e)}")
            return text

class ImagePreprocessor:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    async def preprocess_image(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Create masks for printed and handwritten text
            printed_mask = await self._create_printed_text_mask(gray)
            handwritten_mask = await self._create_handwritten_mask(gray)
            
            # Process regions
            printed_regions = cv2.bitwise_and(gray, gray, mask=printed_mask)
            printed_regions = await self._enhance_printed_text(printed_regions)
            
            handwritten_regions = cv2.bitwise_and(gray, gray, mask=handwritten_mask)
            handwritten_regions = await self._enhance_handwritten_text(handwritten_regions)
            
            return {
                'printed': printed_regions,
                'handwritten': handwritten_regions
            }
        except Exception as e:
            self.logger.error(f"Image preprocessing error: {str(e)}")
            raise

    async def _create_printed_text_mask(self, gray_image: np.ndarray) -> np.ndarray:
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((3,3), np.uint8)
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    async def _create_handwritten_mask(self, gray_image: np.ndarray) -> np.ndarray:
        binary = cv2.adaptiveThreshold(
            gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        kernel = np.ones((2,2), np.uint8)
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    async def _enhance_printed_text(self, image: np.ndarray) -> np.ndarray:
        denoised = cv2.fastNlMeansDenoising(image)
        return cv2.convertScaleAbs(denoised, alpha=1.2, beta=0)

    async def _enhance_handwritten_text(self, image: np.ndarray) -> np.ndarray:
        filtered = cv2.bilateralFilter(image, 9, 75, 75)
        return cv2.convertScaleAbs(filtered, alpha=1.3, beta=10)

class HomeworkAnalyzer:
    def __init__(self, bulgarian_processor: BulgarianTextProcessor):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.bulgarian_processor = bulgarian_processor
        self.initialize_models()

    def initialize_models(self):
        try:
            self.answer_model = T5ForConditionalGeneration.from_pretrained('t5-base')
            self.answer_tokenizer = T5Tokenizer.from_pretrained('t5-base')
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
                'nlptown/bert-base-multilingual-uncased-sentiment'
            )
            self.sentiment_tokenizer = AutoTokenizer.from_pretrained(
                'nlptown/bert-base-multilingual-uncased-sentiment'
            )
        except Exception as e:
            self.logger.error(f"Model initialization error: {str(e)}")
            raise

    async def analyze_homework(
        self, 
        extracted_text: Dict[str, str],
        subject: Optional[str] = None,
        rubric: Optional[Dict] = None
    ) -> Dict:
        try:
            # Translate texts
            question_en = await self.bulgarian_processor.translate_to_english(
                extracted_text['printed']
            )
            answer_en = await self.bulgarian_processor.translate_to_english(
                extracted_text['handwritten']
            )
            
            # Analyze content
            analysis_result = await self._analyze_content(
                question_en,
                answer_en,
                subject,
                rubric
            )
            
            # Generate feedback
            feedback_en = await self._generate_feedback(analysis_result)
            feedback_bg = await self.bulgarian_processor.translate_to_bulgarian(feedback_en)
            
            return {
                'extracted_text': extracted_text,
                'analysis': analysis_result,
                'feedback_bulgarian': feedback_bg
            }
        except Exception as e:
            self.logger.error(f"Analysis error: {str(e)}")
            raise

    async def _analyze_content(
        self,
        question: str,
        answer: str,
        subject: Optional[str],
        rubric: Optional[Dict]
    ) -> Dict:
        try:
            # Prepare analysis prompt
            prompt = self._create_analysis_prompt(question, answer, subject, rubric)
            
            # Generate analysis
            inputs = self.answer_tokenizer(
                prompt, 
                max_length=512,
                truncation=True,
                return_tensors="pt"
            )
            
            outputs = self.answer_model.generate(
                inputs.input_ids,
                max_length=200,
                num_beams=4,
                early_stopping=True
            )
            
            analysis = self.answer_tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            # Parse and structure the analysis
            result = await self._structure_analysis(analysis, subject, rubric)
            
            # Add sentiment analysis
            result['sentiment'] = await self._analyze_sentiment(answer)
            
            return result
        except Exception as e:
            self.logger.error(f"Content analysis error: {str(e)}")
            raise

    def _create_analysis_prompt(
        self,
        question: str,
        answer: str,
        subject: Optional[str],
        rubric: Optional[Dict]
    ) -> str:
        prompt = f"""
        Question: {question}
        Answer: {answer}
        Subject: {subject if subject else 'general'}
        
        Analyze the answer considering:
        1. Completeness and accuracy
        2. Understanding of concepts
        3. Clarity of explanation
        4. Use of subject-specific terminology
        5. Overall quality
        
        {self._format_rubric_criteria(rubric) if rubric else ''}
        """
        return prompt

    def _format_rubric_criteria(self, rubric: Dict) -> str:
        if not rubric:
            return ""
        
        criteria = ["Additional criteria to consider:"]
        for criterion, weight in rubric.items():
            criteria.append(f"- {criterion} (weight: {weight})")
        return "\n".join(criteria)

    async def _structure_analysis(
        self,
        analysis: str,
        subject: Optional[str],
        rubric: Optional[Dict]
    ) -> Dict:
        result = {
            'completeness': 0.0,
            'understanding': 0.0,
            'clarity': 0.0,
            'terminology': 0.0,
            'overall_quality': 0.0,
            'key_points': [],
            'missing_points': [],
            'suggestions': []
        }
        
        # Extract scores and points from analysis text
        lines = analysis.split('\n')
        current_section = ''
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.lower().strip()
                value = value.strip()
                
                if key in result:
                    try:
                        result[key] = float(value.rstrip('%')) / 100
                    except:
                        continue
                        
            elif line.startswith('- '):
                if current_section:
                    result[f'{current_section}_points'].append(line[2:])
            elif line.lower() in ['key points', 'missing points', 'suggestions']:
                current_section = line.lower().split()[0]
        
        # Apply rubric weights if provided
        if rubric:
            weighted_score = 0.0
            total_weight = sum(rubric.values())
            
            for criterion, weight in rubric.items():
                if criterion in result:
                    weighted_score += result[criterion] * (weight / total_weight)
            
            result['weighted_score'] = weighted_score
        
        return result

    async def _analyze_sentiment(self, text: str) -> Dict:
        try:
            inputs = self.sentiment_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            outputs = self.sentiment_model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=1)
            
            return {
                'score': float(scores.max()),
                'label': str(scores.argmax().item() + 1)  # 1-5 scale
            }
        except Exception as e:
            self.logger.error(f"Sentiment analysis error: {str(e)}")
            return {'score': 0.0, 'label': '3'}

    async def _generate_feedback(self, analysis: Dict) -> str:
        try:
            # Generate detailed feedback based on analysis results
            feedback_points = []
            
            # Overall assessment
            overall_score = analysis.get('overall_quality', 0)
            if overall_score >= 0.8:
                feedback_points.append("Excellent work! The response demonstrates strong understanding.")
            elif overall_score >= 0.6:
                feedback_points.append("Good work with room for improvement.")
            else:
                feedback_points.append("The response needs significant improvement.")
            
            # Specific strengths
            if analysis.get('key_points'):
                feedback_points.append("\nStrengths:")
                for point in analysis['key_points']:
                    feedback_points.append(f"- {point}")
            
            # Areas for improvement
            if analysis.get('missing_points'):
                feedback_points.append("\nAreas for improvement:")
                for point in analysis['missing_points']:
                    feedback_points.append(f"- {point}")
            
            # Suggestions
            if analysis.get('suggestions'):
                feedback_points.append("\nSuggestions:")
                for suggestion in analysis['suggestions']:
                    feedback_points.append(f"- {suggestion}")
            
            return "\n".join(feedback_points)
        except Exception as e:
            self.logger.error(f"Feedback generation error: {str(e)}")
            return "Unable to generate detailed feedback."

class HomeworkAnalysisSystem:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.bulgarian_processor = BulgarianTextProcessor()
        self.image_preprocessor = ImagePreprocessor()
        self.analyzer = HomeworkAnalyzer(self.bulgarian_processor)

    async def process_homework(
        self,
        image_data: bytes,
        subject: Optional[str] = None,
        rubric: Optional[Dict] = None
    ) -> ProcessingResult:
        try:
            # Convert image bytes to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Preprocess image
            processed_images = await self.image_preprocessor.preprocess_image(image)
            
            # Extract text using OCR
            extracted_text = {
                'printed': await self._extract_text(processed_images['printed'], 'printed'),
                'handwritten': await self._extract_text(processed_images['handwritten'], 'handwritten')
            }
            
            # Analyze homework
            analysis_results = await self.analyzer.analyze_homework(
                extracted_text,
                subject,
                rubric
            )
            
            return ProcessingResult(
                success=True,
                message="Analysis completed successfully",
                data=analysis_results
            )
        except Exception as e:
            self.logger.error(f"Processing error: {str(e)}")
            return ProcessingResult(
                success=False,
                message=f"Processing failed: {str(e)}"
            )

    async def _extract_text(self, image: np.ndarray, text_type: str) -> str:
        try:
            if text_type == 'printed':
                # Use Tesseract for printed text
                custom_config = r'--oem 3 --psm 6 -l bul'
                text = pytesseract.image_to_string(image, config=custom_config)
            else:
                # Use EasyOCR for handwritten text
                results = self.bulgarian_processor.reader.readtext(image)
                text = " ".join([result[1] for result in results])
            
            return text.strip()
        except Exception as e:
            self.logger.error(f"Text extraction error: {str(e)}")
            return ""

# FastAPI application setup
app = FastAPI(title="AI Homework Analysis System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize system
system = HomeworkAnalysisSystem()

class AnalysisRequest(BaseModel):
    subject: Optional[str] = None
    rubric: Optional[Dict] = None

class AnalysisResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict] = None

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_homework(
    file: UploadFile = File(...),
    request: AnalysisRequest = None
):
    try:
        contents = await file.read()
        result = await system.process_homework(
            contents,
            request.subject if request else None,
            request.rubric if request else None
        )
        
        return AnalysisResponse(
            success=result.success,
            message=result.message,
            data=result.data
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-batch", response_model=AnalysisResponse)
async def analyze_batch_homework(
    files: List[UploadFile] = File(...),
    request: AnalysisRequest = None,
    background_tasks: BackgroundTasks = None
):
    try:
        results = []
        for file in files:
            contents = await file.read()
            result = await system.process_homework(
                contents,
                request.subject if request else None,
                request.rubric if request else None
            )
            results.append({
                'filename': file.filename,
                'analysis': result.data if result.success else None,
                'error': None if result.success else result.message
            })

        # Save results in background
        if background_tasks:
            background_tasks.add_task(_save_batch_results, results)

        return AnalysisResponse(
            success=True,
            message=f"Processed {len(files)} files",
            data={'results': results}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def _save_batch_results(results: List[Dict]):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f"batch_results_{timestamp}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Error saving batch results: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
