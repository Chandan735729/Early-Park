class AudioFeatures {
  final double age;
  final int sex; // 0 = female, 1 = male
  final double testTime;
  final double jitterPercent;
  final double jitterAbs;
  final double jitterRAP;
  final double jitterPPQ5;
  final double jitterDDP;
  final double shimmer;
  final double shimmerDB;
  final double shimmerAPQ3;
  final double shimmerAPQ5;
  final double shimmerAPQ11;
  final double shimmerDDA;
  final double nhr;
  final double hnr;
  final double rpde;
  final double dfa;
  final double ppe;

  AudioFeatures({
    required this.age,
    required this.sex,
    required this.testTime,
    required this.jitterPercent,
    required this.jitterAbs,
    required this.jitterRAP,
    required this.jitterPPQ5,
    required this.jitterDDP,
    required this.shimmer,
    required this.shimmerDB,
    required this.shimmerAPQ3,
    required this.shimmerAPQ5,
    required this.shimmerAPQ11,
    required this.shimmerDDA,
    required this.nhr,
    required this.hnr,
    required this.rpde,
    required this.dfa,
    required this.ppe,
  });

  // Convert to map for API requests
  Map<String, dynamic> toMap() {
    return {
      'age': age,
      'sex': sex,
      'test_time': testTime,
      'Jitter(%)': jitterPercent,
      'Jitter(Abs)': jitterAbs,
      'Jitter:RAP': jitterRAP,
      'Jitter:PPQ5': jitterPPQ5,
      'Jitter:DDP': jitterDDP,
      'Shimmer': shimmer,
      'Shimmer(dB)': shimmerDB,
      'Shimmer:APQ3': shimmerAPQ3,
      'Shimmer:APQ5': shimmerAPQ5,
      'Shimmer:APQ11': shimmerAPQ11,
      'Shimmer:DDA': shimmerDDA,
      'NHR': nhr,
      'HNR': hnr,
      'RPDE': rpde,
      'DFA': dfa,
      'PPE': ppe,
    };
  }

  // Create from JSON response
  factory AudioFeatures.fromMap(Map<String, dynamic> map) {
    return AudioFeatures(
      age: (map['age'] as num).toDouble(),
      sex: map['sex'] as int,
      testTime: (map['test_time'] as num).toDouble(),
      jitterPercent: (map['Jitter(%)'] as num).toDouble(),
      jitterAbs: (map['Jitter(Abs)'] as num).toDouble(),
      jitterRAP: (map['Jitter:RAP'] as num).toDouble(),
      jitterPPQ5: (map['Jitter:PPQ5'] as num).toDouble(),
      jitterDDP: (map['Jitter:DDP'] as num).toDouble(),
      shimmer: (map['Shimmer'] as num).toDouble(),
      shimmerDB: (map['Shimmer(dB)'] as num).toDouble(),
      shimmerAPQ3: (map['Shimmer:APQ3'] as num).toDouble(),
      shimmerAPQ5: (map['Shimmer:APQ5'] as num).toDouble(),
      shimmerAPQ11: (map['Shimmer:APQ11'] as num).toDouble(),
      shimmerDDA: (map['Shimmer:DDA'] as num).toDouble(),
      nhr: (map['NHR'] as num).toDouble(),
      hnr: (map['HNR'] as num).toDouble(),
      rpde: (map['RPDE'] as num).toDouble(),
      dfa: (map['DFA'] as num).toDouble(),
      ppe: (map['PPE'] as num).toDouble(),
    );
  }
}