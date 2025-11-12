import 'dart:math';
import 'package:flutter/foundation.dart';
import '../services/ml_service.dart';
import '../models/audio_features.dart';

enum PredictionState {
  idle,
  processing,
  completed,
  error,
}

class PredictionResult {
  final double riskScore;
  final String riskLevel;
  final String interpretation;
  final DateTime timestamp;
  final Map<String, dynamic> features;

  PredictionResult({
    required this.riskScore,
    required this.riskLevel,
    required this.interpretation,
    required this.timestamp,
    required this.features,
  });
}

class PredictionProvider extends ChangeNotifier {
  final MLService _mlService = MLService();
  
  PredictionState _state = PredictionState.idle;
  PredictionResult? _lastPrediction;
  String? _errorMessage;
  double _processingProgress = 0.0;

  // Getters
  PredictionState get state => _state;
  PredictionResult? get lastPrediction => _lastPrediction;
  String? get errorMessage => _errorMessage;
  double get processingProgress => _processingProgress;
  bool get isProcessing => _state == PredictionState.processing;
  bool get hasResult => _lastPrediction != null;

  // Predict Parkinson's risk from audio file
  Future<void> predictFromAudio(String audioPath) async {
    try {
      _state = PredictionState.processing;
      _errorMessage = null;
      _processingProgress = 0.0;
      notifyListeners();

      // Complete prediction in one API call with URL fallback
      _processingProgress = 0.2;
      notifyListeners();
      
      debugPrint('🚀 Starting complete prediction from provider...');
      
      // Generate random age between 18-80 for realistic testing
      final random = Random();
      final randomAge = 18 + random.nextInt(63); // 18-80 years
      final randomSex = random.nextInt(2); // 0 or 1
      
      debugPrint('📝 Using age: $randomAge, sex: $randomSex');
      final completeResult = await _mlService.predictComplete(audioPath, age: randomAge.toDouble(), sex: randomSex);
      debugPrint('📊 API Response received: ${completeResult.toString()}');
      
      _processingProgress = 0.9;
      notifyListeners();
      
      // Extract features from the complete result
      final featuresData = completeResult['features'] as Map<String, dynamic>;
      final features = AudioFeatures.fromMap(featuresData);
      
      // Create prediction result
      _lastPrediction = _createPredictionResult(completeResult, features);
      _processingProgress = 1.0;
      _state = PredictionState.completed;
      notifyListeners();
      
    } catch (e) {
      _errorMessage = 'Prediction failed: $e';
      _state = PredictionState.error;
      _processingProgress = 0.0;
      debugPrint('Prediction error: $e');
      notifyListeners();
    }
  }

  // Create prediction result with interpretation
  PredictionResult _createPredictionResult(
    Map<String, dynamic> apiResponse, 
    AudioFeatures features
  ) {
    final score = (apiResponse['risk_score'] as num).toDouble();
    
    String riskLevel;
    String interpretation;
    
    if (score < 15.0) {
      riskLevel = 'Low Risk';
      interpretation = 'Voice patterns suggest low likelihood of Parkinson\'s symptoms. Continue regular health monitoring.';
    } else if (score < 25.0) {
      riskLevel = 'Moderate Risk';
      interpretation = 'Some voice changes detected that may warrant attention. Consider consulting a healthcare professional.';
    } else if (score < 35.0) {
      riskLevel = 'High Risk';
      interpretation = 'Significant voice pattern changes detected. We recommend consulting a neurologist for further evaluation.';
    } else {
      riskLevel = 'Very High Risk';
      interpretation = 'Voice patterns strongly suggest neurological changes. Please seek immediate medical evaluation.';
    }
    
    return PredictionResult(
      riskScore: score,
      riskLevel: riskLevel,
      interpretation: interpretation,
      timestamp: DateTime.now(),
      features: features.toMap(),
    );
  }

  // Clear results
  void clearResults() {
    _lastPrediction = null;
    _state = PredictionState.idle;
    _errorMessage = null;
    _processingProgress = 0.0;
    notifyListeners();
  }

  // Reset to initial state
  void reset() {
    clearResults();
  }
}