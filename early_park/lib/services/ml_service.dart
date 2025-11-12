import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'package:dio/dio.dart';
import 'package:flutter/foundation.dart';
import '../models/audio_features.dart';

class MLService {
  // Try multiple API URLs for different network configurations
  static const List<String> _baseUrls = [
    'http://localhost:5000/api',   // ADB port forwarding (try first)
    'http://127.0.0.1:5000/api',   // Localhost with ADB
    'http://10.0.2.2:5000/api',    // Android emulator
    'http://192.168.1.100:5000/api', // Your local network IP
  ];
  final Dio _dio = Dio();

  MLService() {
    _dio.options.connectTimeout = const Duration(seconds: 10);
    _dio.options.receiveTimeout = const Duration(seconds: 30);
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
        '${_baseUrls[0]}/extract-features',
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
        '${_baseUrls[0]}/predict',
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
      debugPrint('❌ Prediction API error: $e');
      debugPrint('❌ Error type: ${e.runtimeType}');
      if (e is DioException) {
        debugPrint('❌ Dio error details: ${e.message}');
        debugPrint('❌ Response data: ${e.response?.data}');
        debugPrint('❌ Status code: ${e.response?.statusCode}');
      }
      
      // Fallback: generate mock prediction for development
      if (kDebugMode) {
        debugPrint('⚠️ Using mock prediction due to API error');
        return _generateMockPrediction(features);
      }
      
      throw Exception('Failed to get prediction: $e');
    }
  }

  // Generate mock features for development/testing
  AudioFeatures _generateMockFeatures() {
    final random = Random();
    
    return AudioFeatures(
      age: 18.0 + random.nextDouble() * 62.0, // Random age between 18-80
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

  // Complete prediction pipeline: extract features + predict in one call
  Future<Map<String, dynamic>> predictComplete(String audioPath, {double age = 65, int sex = 0}) async {
    debugPrint('🚀 Starting complete prediction for: $audioPath');
    
    // Try each URL until one works
    for (int i = 0; i < _baseUrls.length; i++) {
      final baseUrl = _baseUrls[i];
      try {
        debugPrint('🔗 Trying API URL ${i+1}/${_baseUrls.length}: $baseUrl');
        
        // Create multipart file for upload
        final formData = FormData.fromMap({
          'audio': await MultipartFile.fromFile(
            audioPath,
            filename: 'recording.wav',
          ),
          'age': age.toString(),
          'sex': sex.toString(),
        });

        final response = await _dio.post(
          '$baseUrl/predict-complete',
          data: formData,
          options: Options(
            headers: {'Content-Type': 'multipart/form-data'},
          ),
        );

        debugPrint('✅ API Response Status: ${response.statusCode} from $baseUrl');
        debugPrint('✅ API Response Data: ${response.data}');

        if (response.statusCode == 200 && response.data['success']) {
          debugPrint('🎉 Successfully connected to API at: $baseUrl');
          return response.data;
        } else {
          throw Exception('Complete prediction failed: ${response.data['error']}');
        }
        
      } catch (e) {
        debugPrint('❌ URL ${i+1} failed ($baseUrl): $e');
        if (i == _baseUrls.length - 1) {
          // This was the last URL, fall through to mock data
          break;
        }
        // Continue to next URL
      }
    }

    // If we get here, all URLs failed - use fallback
    debugPrint('🚨 All API URLs failed, using mock prediction');
    if (kDebugMode) {
      debugPrint('⚠️ Using mock prediction while debugging audio format');
      final mockFeatures = _generateMockFeatures();
      final mockPrediction = _generateMockPrediction(mockFeatures);
      // Add features to match API response format
      mockPrediction['features'] = mockFeatures.toMap();
      return mockPrediction;
    }
    
    throw Exception('Failed to connect to any API endpoint - check if Python server is running');
  }

  // Health check for API
  Future<bool> checkApiHealth() async {
    try {
      final response = await _dio.get('${_baseUrls[0]}/health');
      return response.statusCode == 200;
    } catch (e) {
      debugPrint('API health check failed: $e');
      return false;
    }
  }
}