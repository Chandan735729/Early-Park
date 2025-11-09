import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:intl/intl.dart';
import '../providers/prediction_provider.dart';

class ResultDisplay extends StatelessWidget {
  final PredictionResult result;

  const ResultDisplay({
    super.key,
    required this.result,
  });

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      child: Column(
        children: [
          // Main Result Card
          _buildMainResultCard(context),
          
          const SizedBox(height: 20),
          
          // Details Card
          _buildDetailsCard(context),
          
          const SizedBox(height: 20),
          
          // Recommendations Card
          _buildRecommendationsCard(context),
        ],
      ),
    );
  }

  Widget _buildMainResultCard(BuildContext context) {
    return Card(
      elevation: 8,
      child: Container(
        padding: const EdgeInsets.all(24),
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(12),
          gradient: LinearGradient(
            colors: [
              _getRiskColor().withOpacity(0.1),
              _getRiskColor().withOpacity(0.05),
            ],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
        ),
        child: Column(
          children: [
            // Risk Level Badge
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              decoration: BoxDecoration(
                color: _getRiskColor(),
                borderRadius: BorderRadius.circular(20),
              ),
              child: Text(
                result.riskLevel,
                style: const TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                  fontSize: 16,
                ),
              ),
            ),
            
            const SizedBox(height: 20),
            
            // Risk Score Display
            _buildRiskScoreDisplay(context),
            
            const SizedBox(height: 20),
            
            // Interpretation
            Text(
              result.interpretation,
              style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                height: 1.5,
              ),
              textAlign: TextAlign.center,
            ),
            
            const SizedBox(height: 16),
            
            // Timestamp
            Text(
              'Analysis completed: ${DateFormat('MMM dd, yyyy • HH:mm').format(result.timestamp)}',
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                color: Theme.of(context).colorScheme.onSurface.withOpacity(0.6),
              ),
            ),
          ],
        ),
      ),
    ).animate().fadeIn(duration: 600.ms).slideY(
      begin: 0.3,
      end: 0,
      duration: 600.ms,
    );
  }

  Widget _buildRiskScoreDisplay(BuildContext context) {
    return Column(
      children: [
        // Circular Progress Indicator
        SizedBox(
          width: 120,
          height: 120,
          child: Stack(
            alignment: Alignment.center,
            children: [
              CircularProgressIndicator(
                value: (result.riskScore / 50.0).clamp(0.0, 1.0),
                strokeWidth: 12,
                backgroundColor: Colors.grey.withOpacity(0.3),
                valueColor: AlwaysStoppedAnimation<Color>(_getRiskColor()),
              ),
              Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Text(
                    result.riskScore.toStringAsFixed(1),
                    style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                      fontWeight: FontWeight.bold,
                      color: _getRiskColor(),
                    ),
                  ),
                  Text(
                    'UPDRS Score',
                    style: Theme.of(context).textTheme.bodySmall?.copyWith(
                      color: Theme.of(context).colorScheme.onSurface.withOpacity(0.7),
                    ),
                  ),
                ],
              ),
            ],
          ),
        ).animate().scale(
          begin: const Offset(0.8, 0.8),
          end: const Offset(1.0, 1.0),
          duration: 800.ms,
        ),
      ],
    );
  }

  Widget _buildDetailsCard(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(
                  Icons.analytics,
                  color: Theme.of(context).colorScheme.primary,
                ),
                const SizedBox(width: 12),
                Text(
                  'Analysis Details',
                  style: Theme.of(context).textTheme.titleLarge?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
            
            const SizedBox(height: 20),
            
            // Key Features Display
            _buildFeatureItem(
              context,
              'Voice Jitter',
              '${(result.features['Jitter(%)'] * 100).toStringAsFixed(3)}%',
              Icons.graphic_eq,
            ),
            
            _buildFeatureItem(
              context,
              'Voice Shimmer',
              '${(result.features['Shimmer'] * 100).toStringAsFixed(2)}%',
              Icons.waves,
            ),
            
            _buildFeatureItem(
              context,
              'Harmony-to-Noise Ratio',
              '${result.features['HNR'].toStringAsFixed(1)} dB',
              Icons.tune,
            ),
            
            _buildFeatureItem(
              context,
              'Age Factor',
              '${result.features['age'].toStringAsFixed(0)} years',
              Icons.person,
            ),
          ],
        ),
      ),
    ).animate().fadeIn(delay: 200.ms, duration: 600.ms);
  }

  Widget _buildFeatureItem(
    BuildContext context,
    String label,
    String value,
    IconData icon,
  ) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Row(
        children: [
          Icon(
            icon,
            size: 20,
            color: Theme.of(context).colorScheme.onSurface.withOpacity(0.7),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              label,
              style: Theme.of(context).textTheme.bodyMedium,
            ),
          ),
          Text(
            value,
            style: Theme.of(context).textTheme.bodyMedium?.copyWith(
              fontWeight: FontWeight.bold,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildRecommendationsCard(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(
                  Icons.lightbulb_outline,
                  color: Theme.of(context).colorScheme.secondary,
                ),
                const SizedBox(width: 12),
                Text(
                  'Recommendations',
                  style: Theme.of(context).textTheme.titleLarge?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
            
            const SizedBox(height: 16),
            
            ...(_getRecommendations().map((rec) => _buildRecommendationItem(context, rec))),
            
            const SizedBox(height: 16),
            
            // Disclaimer
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Theme.of(context).colorScheme.surfaceVariant,
                borderRadius: BorderRadius.circular(8),
              ),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Icon(
                    Icons.info_outline,
                    size: 16,
                    color: Theme.of(context).colorScheme.onSurfaceVariant,
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      'This analysis is for screening purposes only and should not replace professional medical evaluation.',
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                        color: Theme.of(context).colorScheme.onSurfaceVariant,
                        fontStyle: FontStyle.italic,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    ).animate().fadeIn(delay: 400.ms, duration: 600.ms);
  }

  Widget _buildRecommendationItem(BuildContext context, String recommendation) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 6,
            height: 6,
            margin: const EdgeInsets.only(top: 8, right: 12),
            decoration: BoxDecoration(
              color: Theme.of(context).colorScheme.primary,
              shape: BoxShape.circle,
            ),
          ),
          Expanded(
            child: Text(
              recommendation,
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                height: 1.4,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Color _getRiskColor() {
    if (result.riskScore < 15.0) {
      return Colors.green;
    } else if (result.riskScore < 25.0) {
      return Colors.orange;
    } else if (result.riskScore < 35.0) {
      return Colors.deepOrange;
    } else {
      return Colors.red;
    }
  }

  List<String> _getRecommendations() {
    if (result.riskScore < 15.0) {
      return [
        'Continue regular health check-ups',
        'Maintain a healthy lifestyle with regular exercise',
        'Consider annual voice screenings if you have risk factors',
      ];
    } else if (result.riskScore < 25.0) {
      return [
        'Schedule a consultation with your primary care physician',
        'Monitor for other early symptoms of movement disorders',
        'Consider lifestyle modifications for brain health',
        'Regular follow-up voice screenings recommended',
      ];
    } else if (result.riskScore < 35.0) {
      return [
        'Seek evaluation from a neurologist specializing in movement disorders',
        'Document any other symptoms you may have noticed',
        'Consider bringing a family member to medical appointments',
        'Ask about early intervention strategies',
      ];
    } else {
      return [
        'Schedule an urgent consultation with a neurologist',
        'Bring this analysis result to your healthcare provider',
        'Document all symptoms and their progression',
        'Consider seeking a second opinion if needed',
        'Explore support resources for patients and families',
      ];
    }
  }
}