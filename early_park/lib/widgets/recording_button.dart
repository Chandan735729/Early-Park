import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import '../providers/audio_provider.dart';
import '../providers/prediction_provider.dart';

class RecordingButton extends StatelessWidget {
  final AudioProvider audioProvider;
  final PredictionProvider predictionProvider;

  const RecordingButton({
    super.key,
    required this.audioProvider,
    required this.predictionProvider,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        // Main Recording Button
        GestureDetector(
          onTap: _handleButtonTap,
          child: Container(
            width: 200,
            height: 200,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: _getButtonColor(context),
              boxShadow: [
                BoxShadow(
                  color: _getButtonColor(context).withOpacity(0.3),
                  blurRadius: 20,
                  spreadRadius: audioProvider.isRecording ? 10 : 0,
                ),
              ],
            ),
            child: Stack(
              alignment: Alignment.center,
              children: [
                // Pulsing animation when recording
                if (audioProvider.isRecording)
                  Container(
                    width: 180,
                    height: 180,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      border: Border.all(
                        color: Colors.white.withOpacity(0.5),
                        width: 2,
                      ),
                    ),
                  ).animate(onPlay: (controller) => controller.repeat())
                    .scale(
                      begin: const Offset(0.9, 0.9),
                      end: const Offset(1.1, 1.1),
                      duration: 1500.ms,
                    )
                    .then()
                    .scale(
                      begin: const Offset(1.1, 1.1),
                      end: const Offset(0.9, 0.9),
                      duration: 1500.ms,
                    ),
                
                // Icon
                Icon(
                  _getButtonIcon(),
                  size: 80,
                  color: Colors.white,
                ),
              ],
            ),
          ),
        ),
        
        const SizedBox(height: 24),
        
        // Status Text
        Text(
          _getStatusText(),
          style: Theme.of(context).textTheme.titleMedium?.copyWith(
            fontWeight: FontWeight.w600,
            color: Theme.of(context).colorScheme.onSurface,
          ),
          textAlign: TextAlign.center,
        ),
        
        // Duration Display
        if (audioProvider.isRecording)
          Padding(
            padding: const EdgeInsets.only(top: 12),
            child: Text(
              _formatDuration(audioProvider.recordingDuration),
              style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                fontWeight: FontWeight.bold,
                color: Theme.of(context).colorScheme.primary,
              ),
            ).animate(onPlay: (controller) => controller.repeat())
              .fadeIn(duration: 500.ms)
              .then()
              .fadeOut(duration: 500.ms),
          ),
        
        // Control Buttons Row
        if (audioProvider.hasRecording && !audioProvider.isRecording)
          Padding(
            padding: const EdgeInsets.only(top: 24),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                // Delete Button
                _buildControlButton(
                  context: context,
                  icon: Icons.delete_outline,
                  label: 'Delete',
                  onTap: audioProvider.deleteRecording,
                  color: Theme.of(context).colorScheme.error,
                ),
                
                const SizedBox(width: 24),
                
                // Analyze Button
                _buildControlButton(
                  context: context,
                  icon: Icons.analytics_outlined,
                  label: 'Analyze',
                  onTap: _handleAnalyze,
                  color: Theme.of(context).colorScheme.primary,
                  isPrimary: true,
                ),
              ],
            ),
          ),
      ],
    );
  }

  Widget _buildControlButton({
    required BuildContext context,
    required IconData icon,
    required String label,
    required VoidCallback onTap,
    required Color color,
    bool isPrimary = false,
  }) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
        decoration: BoxDecoration(
          color: isPrimary ? color : Colors.transparent,
          border: Border.all(color: color, width: 2),
          borderRadius: BorderRadius.circular(25),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              icon,
              color: isPrimary ? Colors.white : color,
              size: 20,
            ),
            const SizedBox(width: 8),
            Text(
              label,
              style: TextStyle(
                color: isPrimary ? Colors.white : color,
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Color _getButtonColor(BuildContext context) {
    switch (audioProvider.recordingState) {
      case RecordingState.recording:
        return Colors.red;
      case RecordingState.stopped:
        return Theme.of(context).colorScheme.primary;
      case RecordingState.error:
        return Theme.of(context).colorScheme.error;
      default:
        return Theme.of(context).colorScheme.primary;
    }
  }

  IconData _getButtonIcon() {
    switch (audioProvider.recordingState) {
      case RecordingState.recording:
        return Icons.stop;
      case RecordingState.stopped:
        return Icons.mic;
      case RecordingState.error:
        return Icons.error;
      default:
        return Icons.mic;
    }
  }

  String _getStatusText() {
    if (predictionProvider.isProcessing) {
      return 'Analyzing voice patterns...';
    }
    
    switch (audioProvider.recordingState) {
      case RecordingState.recording:
        return 'Recording... Tap to stop';
      case RecordingState.stopped:
        return audioProvider.hasRecording 
            ? 'Recording complete. Ready to analyze!'
            : 'Tap to start recording';
      case RecordingState.error:
        return 'Recording failed. Try again.';
      default:
        return 'Tap to start voice recording';
    }
  }

  String _formatDuration(Duration duration) {
    final minutes = duration.inMinutes.toString().padLeft(2, '0');
    final seconds = (duration.inSeconds % 60).toString().padLeft(2, '0');
    return '$minutes:$seconds';
  }

  void _handleButtonTap() {
    if (predictionProvider.isProcessing) return;
    
    if (audioProvider.isRecording) {
      audioProvider.stopRecording();
    } else {
      predictionProvider.clearResults();
      audioProvider.startRecording();
    }
  }

  void _handleAnalyze() {
    if (audioProvider.recordingPath != null) {
      predictionProvider.predictFromAudio(audioProvider.recordingPath!);
    }
  }
}