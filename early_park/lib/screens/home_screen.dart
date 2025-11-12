import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:flutter_animate/flutter_animate.dart';
import '../providers/audio_provider.dart';
import '../providers/prediction_provider.dart';
import '../widgets/recording_button.dart';
import '../widgets/result_display.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      context.read<AudioProvider>().initializePermissions();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Theme.of(context).colorScheme.background,
      appBar: AppBar(
        elevation: 0,
        backgroundColor: Colors.transparent,
        title: Text(
          'EarlyPark',
          style: TextStyle(
            fontWeight: FontWeight.bold,
            color: Theme.of(context).colorScheme.primary,
          ),
        ),
        centerTitle: true,
        actions: [
          IconButton(
            onPressed: () => _showInfoDialog(context),
            icon: Icon(
              Icons.info_outline,
              color: Theme.of(context).colorScheme.primary,
            ),
          ),
        ],
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          child: Padding(
            padding: const EdgeInsets.all(24.0),
            child: Column(
              children: [
                // Header Section
                _buildHeaderSection(context),

                const SizedBox(height: 40),

                // Recording Section
                Consumer2<AudioProvider, PredictionProvider>(
                  builder: (context, audioProvider, predictionProvider, child) {
                    return Column(
                      children: [
                        // Recording Button
                        RecordingButton(
                              audioProvider: audioProvider,
                              predictionProvider: predictionProvider,
                            )
                            .animate()
                            .fadeIn(duration: 600.ms)
                            .scale(
                              begin: const Offset(0.8, 0.8),
                              end: const Offset(1.0, 1.0),
                              duration: 600.ms,
                            ),

                        const SizedBox(height: 40),

                        // Status and Results
                        if (predictionProvider.isProcessing)
                          _buildProcessingIndicator(predictionProvider),

                        if (predictionProvider.hasResult)
                          ResultDisplay(
                                    result:
                                        predictionProvider.lastPrediction!,
                                  )
                                  .animate()
                                  .fadeIn(duration: 800.ms)
                                  .slideY(
                                    begin: 0.3,
                                    end: 0,
                                    duration: 800.ms,
                                  ),

                        if (audioProvider.errorMessage != null)
                          _buildErrorCard(audioProvider.errorMessage!),
                      ],
                    );
                  },
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildHeaderSection(BuildContext context) {
    return Column(
      children: [
        Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                color: Theme.of(context).colorScheme.primaryContainer,
                borderRadius: BorderRadius.circular(20),
              ),
              child: Column(
                children: [
                  Icon(
                    Icons.mic_rounded,
                    size: 60,
                    color: Theme.of(context).colorScheme.primary,
                  ),
                  const SizedBox(height: 16),
                  Text(
                    'Voice-Based Parkinson\'s Detection',
                    style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                      fontWeight: FontWeight.bold,
                      color: Theme.of(context).colorScheme.onPrimaryContainer,
                    ),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 12),
                  Text(
                    'Record your voice for AI-powered early detection analysis',
                    style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                      color: Theme.of(
                        context,
                      ).colorScheme.onPrimaryContainer.withOpacity(0.8),
                    ),
                    textAlign: TextAlign.center,
                  ),
                ],
              ),
            )
            .animate()
            .fadeIn(duration: 800.ms)
            .slideY(begin: -0.2, end: 0, duration: 800.ms),
      ],
    );
  }

  Widget _buildProcessingIndicator(PredictionProvider predictionProvider) {
    return Card(
      elevation: 4,
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          children: [
            CircularProgressIndicator(
              value: predictionProvider.processingProgress,
              strokeWidth: 6,
            ),
            const SizedBox(height: 16),
            Text(
              'Analyzing voice patterns...',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 8),
            Text(
              '${(predictionProvider.processingProgress * 100).toInt()}% Complete',
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                color: Theme.of(context).colorScheme.onSurface.withOpacity(0.7),
              ),
            ),
          ],
        ),
      ),
    ).animate().fadeIn();
  }

  Widget _buildErrorCard(String error) {
    return Card(
      color: Theme.of(context).colorScheme.errorContainer,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(
          children: [
            Icon(
              Icons.error_outline,
              color: Theme.of(context).colorScheme.error,
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Text(
                error,
                style: TextStyle(
                  color: Theme.of(context).colorScheme.onErrorContainer,
                ),
              ),
            ),
          ],
        ),
      ),
    ).animate().shake(duration: 600.ms);
  }

  void _showInfoDialog(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('About EarlyPark'),
        content: const Text(
          'EarlyPark uses advanced AI to analyze voice patterns that may indicate early signs of Parkinson\'s disease. This tool is for screening purposes only and should not replace professional medical diagnosis.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Got it'),
          ),
        ],
      ),
    );
  }
}
