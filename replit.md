# Personal Finance Assistant

## Overview

This is a machine learning-powered personal finance application that provides expense predictions and financial insights. The system consists of a Python Flask backend that performs ML-based expense analysis and prediction, integrated with a Flutter mobile frontend for user interaction. The application leverages Firebase for authentication and data storage, using Firestore to manage transaction data and user information.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Backend Architecture
- **Framework**: Python Flask web server serving as the main API backend
- **Machine Learning**: Scikit-learn with Linear Regression for expense prediction and pattern analysis
- **Data Processing**: Pandas and NumPy for data manipulation and statistical calculations
- **API Design**: RESTful endpoints with JSON responses, including a `/predict` endpoint for ML predictions

### Authentication & Authorization
- **Firebase Authentication**: Token-based authentication using Firebase Auth ID tokens
- **Authorization Pattern**: Bearer token validation on protected endpoints
- **User Context**: Firebase UID used to scope data access per user

### Data Storage
- **Primary Database**: Google Firestore (NoSQL document database)
- **Data Structure**: Transaction documents stored per user with expense categories, amounts, and timestamps
- **ML Data Pipeline**: Real-time data retrieval from Firestore for model training and predictions

### Frontend Architecture
- **Framework**: Flutter mobile application
- **State Management**: Provider pattern for application state management
- **HTTP Client**: Standard HTTP package for API communication with timeout handling
- **UI Components**: Material Design components with loading states and error handling

### ML Model Architecture
- **Algorithm**: Linear Regression for expense prediction
- **Feature Engineering**: Label encoding for categorical data (expense categories)
- **Data Preprocessing**: Pandas-based data transformation and cleaning
- **Model Persistence**: In-memory model storage with global variables

## External Dependencies

### Firebase Services
- **Firebase Admin SDK**: Server-side Firebase integration for Firestore access
- **Firebase Authentication**: User authentication and token management
- **Firestore Database**: Document-based NoSQL database for transaction storage

### Machine Learning Stack
- **Scikit-learn**: Linear regression model and label encoding
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing and array operations

### Mobile Development
- **Flutter Framework**: Cross-platform mobile app development
- **Provider Package**: State management solution
- **HTTP Package**: Network requests and API communication
- **Intl Package**: Internationalization and date formatting

### Development Environment
- **Replit Hosting**: Cloud-based development and deployment platform
- **Python Flask**: Web framework for API development
- **Firebase Console**: Database and authentication management

### Configuration Requirements
- **Firebase Credentials**: Firebase service account credentials stored as FIREBASE_KEY secret
- **Environment Variables**: API endpoints and configuration settings
- **CORS Configuration**: Cross-origin resource sharing for mobile app integration

## Recent Changes (September 10, 2025)

### API Implementation Completed
- **User Authentication**: All API endpoints now require Firebase Auth Bearer tokens
- **User-Specific Data**: API endpoints now work with user-scoped transaction data
- **Prediction Algorithm**: Implemented next month expense prediction using Linear Regression
- **Firebase Integration**: Updated to use FIREBASE_KEY secret instead of JSON file
- **Response Format**: API responses now match Flutter app expectations with `predicted_expense` field

### API Endpoints
- `GET /` - API information and available endpoints
- `GET /health` - Health check with Firebase connection status
- `GET /transactions` - Get user's transactions (requires auth)
- `GET /train` - Train ML model with user's data (requires auth)
- `POST /predict` - Generate next month expense prediction (requires auth)
- `GET /predictions` - Get stored predictions (requires auth)

### Data Structure
- **User Transactions**: `users/{uid}/transactions/` collection
- **User Predictions**: `users/{uid}/prediction/next_month` document
- **Prediction Format**: `{ predicted_expense: number, created_at: timestamp, month: string }`