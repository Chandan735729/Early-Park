import 'package:flutter/foundation.dart';
import 'package:record/record.dart' as record;

// Mobile-only imports (not available on web)
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as path;
import 'dart:io';

enum RecordingState { idle, recording, stopped, error }

class AudioProvider extends ChangeNotifier {
  final record.AudioRecorder _recorder = record.AudioRecorder();
  RecordingState _recordingState = RecordingState.idle;
  String? _recordingPath;
  Duration _recordingDuration = Duration.zero;
  String? _errorMessage;
  bool _hasPermission = false;

  // Getters
  RecordingState get recordingState => _recordingState;
  String? get recordingPath => _recordingPath;
  Duration get recordingDuration => _recordingDuration;
  String? get errorMessage => _errorMessage;
  bool get hasPermission => _hasPermission;
  bool get isRecording => _recordingState == RecordingState.recording;
  bool get hasRecording => _recordingPath != null;

  // Initialize permissions - Web-compatible
  Future<void> initializePermissions() async {
    debugPrint(
      '🔄 Initializing permissions for platform: ${kIsWeb ? "Web" : "Mobile"}',
    );

    if (kIsWeb) {
      // Web: Browser handles microphone permissions automatically
      debugPrint(
        '🌐 Web: Permission granted - browser will handle microphone access',
      );
      _hasPermission = true;
      _clearError();
    } else {
      // Mobile: For now, assume permission is granted
      // TODO: Add proper mobile permission handling when needed
      debugPrint(
        '📱 Mobile: Permission assumed granted (permission_handler not included)',
      );
      _hasPermission = true;
      _clearError();
    }

    notifyListeners();
  }

  // Start recording
  Future<void> startRecording() async {
    try {
      if (!_hasPermission) {
        await initializePermissions();
        if (!_hasPermission) return;
      }

      _clearError();
      _recordingState = RecordingState.recording;
      notifyListeners();

      // Handle file path and recording differently for web vs mobile
      final fileName =
          'voice_recording_${DateTime.now().millisecondsSinceEpoch}.wav';

      if (kIsWeb) {
        // Web: No path needed - browser handles file storage
        _recordingPath = fileName;
        debugPrint('🌐 Starting web recording: $fileName');
        await _recorder.start(
          const record.RecordConfig(
            encoder: record.AudioEncoder.wav,
            bitRate: 128000,
          ),
          path: "",
        );
      } else {
        // Mobile: Use app directory with full path
        try {
          final directory = await getApplicationDocumentsDirectory();
          _recordingPath = path.join(directory.path, fileName);
          debugPrint('📱 Starting mobile recording: $_recordingPath');
          await _recorder.start(
            const record.RecordConfig(
              encoder: record.AudioEncoder.wav,
              bitRate: 128000,
            ),
            path: _recordingPath!,
          );
        } catch (dirError) {
          debugPrint('📱 Directory error: $dirError');
          rethrow;
        }
      }

      // Start duration timer
      _startDurationTimer();
    } catch (e) {
      _setError('Failed to start recording: $e');
      _recordingState = RecordingState.error;
      notifyListeners();
    }
  }

  // Stop recording
  Future<void> stopRecording() async {
    try {
      if (_recordingState != RecordingState.recording) return;

      await _recorder.stop();
      _recordingState = RecordingState.stopped;

      // Verify recording completed (platform-specific)
      if (_recordingPath != null) {
        if (kIsWeb) {
          debugPrint('🌐 Web recording completed: $_recordingPath');
        } else if (File(_recordingPath!).existsSync()) {
          debugPrint('📱 Recording saved to: $_recordingPath');
          debugPrint(
            '📱 File size: ${File(_recordingPath!).lengthSync()} bytes',
          );
        } else {
          _setError('Recording file was not saved properly');
        }
      }

      notifyListeners();
    } catch (e) {
      _setError('Failed to stop recording: $e');
      _recordingState = RecordingState.error;
      notifyListeners();
    }
  }

  // Delete current recording (platform-specific)
  Future<void> deleteRecording() async {
    try {
      if (_recordingPath != null &&
          !kIsWeb &&
          File(_recordingPath!).existsSync()) {
        await File(_recordingPath!).delete();
        debugPrint('📱 Mobile recording deleted');
      } else if (kIsWeb) {
        debugPrint('🌐 Web recording cleared from memory');
      }

      _recordingPath = null;
      _recordingDuration = Duration.zero;
      _recordingState = RecordingState.idle;
      _clearError();
      notifyListeners();
    } catch (e) {
      _setError('Failed to delete recording: $e');
    }
  }

  // Reset to initial state
  void reset() {
    _recordingState = RecordingState.idle;
    _recordingPath = null;
    _recordingDuration = Duration.zero;
    _clearError();
    notifyListeners();
  }

  // Private methods
  void _startDurationTimer() {
    _recordingDuration = Duration.zero;

    // Update duration every second while recording
    Stream.periodic(const Duration(seconds: 1)).listen((timer) {
      if (_recordingState == RecordingState.recording) {
        _recordingDuration = Duration(
          seconds: _recordingDuration.inSeconds + 1,
        );
        notifyListeners();
      }
    });
  }

  void _setError(String message) {
    _errorMessage = message;
    debugPrint('AudioProvider Error: $message');
  }

  void _clearError() {
    _errorMessage = null;
  }

  @override
  void dispose() {
    _recorder.dispose();
    super.dispose();
  }
}
