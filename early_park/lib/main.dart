import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'screens/home_screen.dart';
import 'providers/audio_provider.dart';
import 'providers/prediction_provider.dart';

void main() {
  runApp(const EarlyParkApp());
}

class EarlyParkApp extends StatelessWidget {
  const EarlyParkApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => AudioProvider()),
        ChangeNotifierProvider(create: (_) => PredictionProvider()),
      ],
      child: MaterialApp(
        title: 'EarlyPark - Parkinson\'s Detection',
        debugShowCheckedModeBanner: false,
        theme: ThemeData(
          useMaterial3: true,
          colorScheme: ColorScheme.fromSeed(
            seedColor: const Color(0xFF2E7D32), // Medical green
            brightness: Brightness.light,
          ),
          fontFamily: 'Roboto',
        ),
        darkTheme: ThemeData(
          useMaterial3: true,
          colorScheme: ColorScheme.fromSeed(
            seedColor: const Color(0xFF2E7D32),
            brightness: Brightness.dark,
          ),
        ),
        home: const HomeScreen(),
      ),
    );
  }
}

