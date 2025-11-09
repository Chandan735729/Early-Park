import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'package:dio/dio.dart';
import 'package:flutter/foundation.dart';
import '../models/audio_features.dart';

class MLService {
  static const String _baseUrl = 'http://localhost:5000/api';
  final Dio _dio = Dio();

  MLService() {
    _dio.options.connectTimeout = const Duration(seconds: 30);
    _dio.options.receiveTimeout = const Duration(seconds: 60);
    _dio.options.headers['Content-Type'] = 'application/json';
  }

  // Extract audio features from WAV file
  Future<AudioFeatures> extractAudioFeatures(String audioPath) async {
    try {
      debugPrint('Extracting features from: $audioPath');
      
      // Create multipart file for upload
      final formData = FormData.fromMap({
        'audio': await MultipartFile.fromFile(
          audioPath,
          filename: 'recording.wav',
        ),
      });

      final response = await _dio.post(
        '$_baseUrl/extract-features',
        data: formData,
        options: Options(
          headers: {'Content-Type': 'multipart/form-data'},
        ),
      );

      if (response.statusCode == 200 && response.data['success']) {
        final featuresData = response.data['features'] as Map<String, dynamic>;
        return AudioFeatures.fromMap(featuresData);
      } else {
        throw Exception('Feature extraction failed: ${response.data['error']}');
      }
      
    } catch (e) {
      debugPrint('Feature extraction error: $e');
      
      // Fallback: generate mock features for development
      if (kDebugMode) {
        debugPrint('Using mock features for development');
        return _generateMockFeatures();
      }
      
      throw Exception('Failed to extract audio features: $e');
    }
  }

  // Send features to ML model for prediction
  Future<Map<String, dynamic>> predictParkinsonRisk(AudioFeatures features) async {
    try {
      debugPrint('Sending features for prediction...');
      
      final response = await _dio.post(
        '$_baseUrl/predict',
        data: json.encode({
          'features': features.toMap(),
        }),
      );

      if (response.statusCode == 200 && response.data['success']) {
        return response.data;
      } else {
        throw Exception('Prediction failed: ${response.data['error']}');
      }
      
    } catch (e) {
      debugPrint('Prediction error: $e');
      
      // Fallback: generate mock prediction for development
      if (kDebugMode) {
        debugPrint('Using mock prediction for development');
        return _generateMockPrediction(features);
      }
      
      throw Exception('Failed to get prediction: $e');
    }
  }

  // Generate mock features for development/testing
  AudioFeatures _generateMockFeatures() {
    final random = Random();
    
    return AudioFeatures(
      age: 65.0 + random.nextDouble() * 20, // Age 65-85
      sex: random.nextInt(2), // 0 or 1
      testTime: random.nextDouble() * 200,
      jitterPercent: 0.003 + random.nextDouble() * 0.01,
      jitterAbs: 0.00002 + random.nextDouble() * 0.00005,
      jitterRAP: 0.001 + random.nextDouble() * 0.005,
      jitterPPQ5: 0.001 + random.nextDouble() * 0.005,
      jitterDDP: 0.003 + random.nextDouble() * 0.015,
      shimmer: 0.015 + random.nextDouble() * 0.05,
      shimmerDB: 0.1 + random.nextDouble() * 0.5,
      shimmerAPQ3: 0.005 + random.nextDouble() * 0.02,
      shimmerAPQ5: 0.007 + random.nextDouble() * 0.03,
      shimmerAPQ11: 0.01 + random.nextDouble() * 0.04,
      shimmerDDA: 0.015 + random.nextDouble() * 0.06,
      nhr: 0.005 + random.nextDouble() * 0.05,
      hnr: 15.0 + random.nextDouble() * 20,
      rpde: 0.3 + random.nextDouble() * 0.5,
      dfa: 0.5 + random.nextDouble() * 0.3,
      ppe: 0.1 + random.nextDouble() * 0.3,
    );
  }

  // Generate mock prediction for development/testing
  Map<String, dynamic> _generateMockPrediction(AudioFeatures features) {
    final random = Random();
    
    // Simulate realistic risk score based on age and some audio features
    double baseRisk = (features.age - 50) * 0.5; // Age factor
    double audioFactor = features.jitterPercent * 1000 + features.shimmer * 100;
    double riskScore = baseRisk + audioFactor + random.nextDouble() * 10;
    
    // Clamp between 0 and 50
    riskScore = riskScore.clamp(0.0, 50.0);
    
    return {
      'success': true,
      'risk_score': riskScore,
      'confidence': 0.85 + random.nextDouble() * 0.1,
      'processing_time': 1.2 + random.nextDouble() * 2.0,
    };
  }

  // Health check for API
  Future<bool> checkApiHealth() async {
    try {
      final response = await _dio.get('$_baseUrl/health');
      return response.statusCode == 200;
    } catch (e) {
      debugPrint('API health check failed: $e');
      return false;
    }
  }
}