import React, { useState } from 'react';
import { 
  Upload, 
  FileText, 
  AlertTriangle, 
  Check, 
  Loader,
  RefreshCw
} from 'lucide-react';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

const SUBJECTS = {
  math: 'Математика',
  literature: 'Литература',
  science: 'Природни науки',
  history: 'История',
  geography: 'География',
  physics: 'Физика',
  chemistry: 'Химия',
  biology: 'Биология'
};

const DEFAULT_RUBRIC = {
  completeness: 0.3,
  understanding: 0.3,
  clarity: 0.2,
  terminology: 0.2
};

const FileUploadArea = ({ onFileSelect, file, loading }) => (
  <div className="space-y-2">
    <Label className="text-lg font-semibold">Качване на домашна работа</Label>
    <div className="border-2 border-dashed border-gray-200 dark:border-gray-800 rounded-lg p-8 text-center">
      <input
        type="file"
        onChange={onFileSelect}
        accept="image/*"
        className="hidden"
        id="file-upload"
        disabled={loading}
      />
      <label htmlFor="file-upload" className="cursor-pointer flex flex-col items-center">
        {!file ? (
          <>
            <Upload className="h-12 w-12 text-gray-400 mb-4" />
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Кликнете за качване или плъзнете файл тук
            </p>
            <p className="text-xs text-gray-500 mt-2">
              Поддържани формати: JPG, PNG, PDF
            </p>
          </>
        ) : (
          <>
            <FileText className="h-12 w-12 text-blue-500 mb-4" />
            <p className="text-sm text-blue-600 dark:text-blue-400">
              Избран файл: {file.name}
            </p>
            <Button 
              variant="ghost" 
              size="sm" 
              className="mt-2"
              onClick={(e) => {
                e.preventDefault();
                document.getElementById('file-upload').value = '';
                onFileSelect({ target: { files: [] } });
              }}
            >
              Изберете друг файл
            </Button>
          </>
        )}
      </label>
    </div>
  </div>
);

const BatchUploader = ({ onUpload, loading }) => {
  const [files, setFiles] = useState([]);

  return (
    <div className="space-y-4">
      <div className="border-2 border-dashed border-gray-200 dark:border-gray-800 rounded-lg p-8">
        <input
          type="file"
          multiple
          onChange={(e) => setFiles(Array.from(e.target.files))}
          accept="image/*"
          className="hidden"
          id="batch-upload"
          disabled={loading}
        />
        <label htmlFor="batch-upload" className="cursor-pointer block text-center">
          <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Изберете множество файлове
          </p>
        </label>
        {files.length > 0 && (
          <div className="mt-4 space-y-2">
            <p className="text-sm text-blue-600 dark:text-blue-400">
              Избрани файлове: {files.length}
            </p>
            <div className="max-h-40 overflow-y-auto">
              {files.map((file, index) => (
                <div key={index} className="flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-800 rounded-md">
                  <span className="text-sm truncate">{file.name}</span>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => {
                      const newFiles = [...files];
                      newFiles.splice(index, 1);
                      setFiles(newFiles);
                    }}
                  >
                    Премахни
                  </Button>
                </div>
              ))}
            </div>
            <Button 
              onClick={() => {
                onUpload(files);
                setFiles([]);
              }}
              className="w-full"
              disabled={loading}
            >
              {loading ? (
                <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                'Обработи файлове'
              )}
            </Button>
          </div>
        )}
      </div>
    </div>
  );
};

const ResultsDisplay = ({ results }) => {
  if (!results) return null;

  const { extracted_text, analysis, feedback_bulgarian } = results;

  return (
    <div className="space-y-6">
      <Alert>
        <Check className="h-4 w-4" />
        <AlertDescription>
          Анализът е завършен успешно
        </AlertDescription>
      </Alert>

      <Tabs defaultValue="text" className="space-y-4">
        <TabsList>
          <TabsTrigger value="text">Извлечен текст</TabsTrigger>
          <TabsTrigger value="analysis">Анализ</TabsTrigger>
          <TabsTrigger value="feedback">Обратна връзка</TabsTrigger>
        </TabsList>

        <TabsContent value="text">
          <Card>
            <CardHeader>
              <CardTitle>Извлечен текст</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <h3 className="font-medium mb-2">Задача</h3>
                <div className="bg-gray-50 dark:bg-gray-900 rounded-md p-4">
                  <p className="whitespace-pre-wrap">{extracted_text.printed}</p>
                </div>
              </div>
              <div>
                <h3 className="font-medium mb-2">Отговор</h3>
                <div className="bg-gray-50 dark:bg-gray-900 rounded-md p-4">
                  <p className="whitespace-pre-wrap">{extracted_text.handwritten}</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="analysis">
          <Card>
            <CardHeader>
              <CardTitle>Анализ на отговора</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-4">
                {Object.entries({
                  'Пълнота': analysis.completeness,
                  'Разбиране': analysis.understanding,
                  'Яснота': analysis.clarity,
                  'Терминология': analysis.terminology,
                  'Обща оценка': analysis.overall_quality
                }).map(([label, value]) => (
                  <div key={label}>
                    <Label>{label}</Label>
                    <Progress value={value * 100} className="mt-2" />
                    <p className="text-sm text-gray-500 mt-1">
                      {Math.round(value * 100)}%
                    </p>
                  </div>
                ))}

                {analysis.key_points.length > 0 && (
                  <div>
                    <Label>Силни страни</Label>
                    <ul className="list-disc list-inside mt-2 space-y-1">
                      {analysis.key_points.map((point, index) => (
                        <li key={index} className="text-sm">{point}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {analysis.missing_points.length > 0 && (
                  <div>
                    <Label>За подобрение</Label>
                    <ul className="list-disc list-inside mt-2 space-y-1">
                      {analysis.missing_points.map((point, index) => (
                        <li key={index} className="text-sm text-amber-600">{point}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="feedback">
          <Card>
            <CardHeader>
              <CardTitle>Обратна връзка</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="prose dark:prose-invert">
                <p className="whitespace-pre-wrap">{feedback_bulgarian}</p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

const HomeworkAnalysis = () => {
  const [file, setFile] = useState(null);
  const [subject, setSubject] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);
  const [mode, setMode] = useState('single');

  const handleFileSelect = (event) => {
    const selectedFile = event.target.files[0];
    setFile(selectedFile);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!file || !subject) return;

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('request', JSON.stringify({
        subject,
        rubric: DEFAULT_RUBRIC
      }));

      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      
      if (data.success) {
        setResults(data.data);
        setFile(null);
      } else {
        setError(data.message);
      }
    } catch (err) {
      setError('Възникна грешка при анализа. Моля, опитайте отново.');
    } finally {
      setLoading(false);
    }
  };

  const handleBatchUpload = async (files) => {
    if (!files.length) return;

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      files.forEach(file => {
        formData.append('files', file);
      });
      formData.append('request', JSON.stringify({
        subject,
        rubric: DEFAULT_RUBRIC
      }));

      const response = await fetch('http://localhost:8000/analyze-batch', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      
      if (data.success) {
        setResults(data.data);
      } else {
        setError(data.message);
      }
    } catch (err) {
      setError('Възникна грешка при обработката. Моля, опитайте отново.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Система за анализ на домашни работи</CardTitle>
          <CardDescription>
            Качете снимка на домашната работа за автоматичен анализ
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            <div className="space-y-2">
              <Label>Изберете предмет</Label>
              <Select value={subject} onValueChange={setSubject}>
                <SelectTrigger>
                  <SelectValue placeholder="Изберете предмет" />
                </SelectTrigger>
                <SelectContent>
                  {Object.entries(SUBJECTS).map(([value, label]) => (
                    <SelectItem key={value} value={value}>
                      {label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <Tabs value={mode} onValueChange={setMode}>
              <TabsList>
                <TabsTrigger value="single">
                  Единичен файл
                </TabsTrigger>
                <TabsTrigger value="batch">
                  Пакетна обработка
                </TabsTrigger>
              </TabsList>

              <TabsContent value="single">
                <FileUploadArea 
                  onFileSelect={handleFileSelect} 
                  file={file}
                  loading={loading}
                />
                <Button
                  onClick={handleAnalyze}
                  className="w-full mt-4"
                  disabled={!file || !subject || loading}
                >
                  {loading ? (
                    <Loader className="mr-2 h-4 w-4 animate-spin" />
                  ) : (
                    'Анализирай'
                  )}
                </Button>
              </TabsContent>

              <TabsContent value="batch">
                <BatchUploader 
                  onUpload={handleBatchUpload}
                  loading={loading}
                />
              </TabsContent>
            </Tabs>

            {error && (
              <Alert variant="destructive">
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}
          </div>
        </CardContent>
      </Card>

      {results && <ResultsDisplay results={results} />}
    </div>
  );
};

export default HomeworkAnalysis;
