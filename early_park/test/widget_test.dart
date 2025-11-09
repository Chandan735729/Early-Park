// This is a basic Flutter widget test.
//
// To perform an interaction with a widget in your test, use the WidgetTester
// utility in the flutter_test package. For example, you can send tap and scroll
// gestures. You can also use WidgetTester to find child widgets in the widget
// tree, read text, and verify that the values of widget properties are correct.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:early_park/main.dart';

void main() {
  testWidgets('EarlyPark app loads correctly', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const EarlyParkApp());

    // Verify that the app title appears
    expect(find.text('EarlyPark'), findsOneWidget);
    
    // Verify that the main UI elements are present
    expect(find.text('Voice-Based Parkinson\'s Detection'), findsOneWidget);
    expect(find.text('Record your voice for AI-powered early detection analysis'), findsOneWidget);
    
    // Verify that the recording interface is present
    // Note: We're just checking if the app builds and shows basic UI
    // Audio recording tests would require more complex setup
  });
}
