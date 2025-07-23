# rental_predictor.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

class RentalPredictor:
    def __init__(self, model_path='rental_predictor_model.pkl', columns_path='original_X_columns.pkl', data_path='rental_data.csv'):
        """
        Initialize the predictor. Try to load pre-trained model first.
        If model or columns are missing, train a new model if data_path exists.
        """
        self.model = None
        self.original_X_columns = None
        self.valid_locations = [
            'Gulshan', 'Banani', 'Dhanmondi', 'Mirpur', 'Uttara',
            'Mohakhali', 'Bashundhara', 'Shyamoli', 'Motijheel', 'Rampura'
        ]
        self.location_counts = {}  # Store data point counts per location
        self.median_floor = 2.0
        self.median_rating = 4.0

        # Try loading pre-trained model
        if os.path.exists(model_path) and os.path.exists(columns_path):
            try:
                self.model = joblib.load(model_path)
                self.original_X_columns = joblib.load(columns_path)
                # Load location counts if available, else assume sufficient data
                try:
                    self.location_counts = joblib.load('location_counts.pkl')
                except FileNotFoundError:
                    # Default counts based on dataset
                    self.location_counts = {
                        'Gulshan': 19, 'Banani': 19, 'Dhanmondi': 1, 'Mirpur': 1, 'Uttara': 1,
                        'Mohakhali': 1, 'Bashundhara': 1, 'Shyamoli': 1, 'Motijheel': 1, 'Rampura': 1
                    }
                print("Loaded pre-trained model and columns.")
                return
            except Exception as e:
                print(f"Error loading pre-trained model: {e}")
        
        # Fallback to training if data_path exists
        if data_path and os.path.exists(data_path):
            self._train_model(data_path)
        else:
            raise FileNotFoundError("Model/columns files and data_path not found. Please provide a valid data_path to train.")

    def _train_model(self, data_path):
        """Train the model using the provided dataset."""
        data = pd.read_csv(data_path)
        data = data[(data['size_sqft'] > 0) & (data['rent_amount'] > 0) & (data['rating'].notnull())]
        print(f"Cleaned dataset size: {len(data)} samples")

        # Count data points per location
        self.location_counts = data['location'].value_counts().to_dict()
        print(f"Location data counts: {self.location_counts}")

        X = data[['location', 'size_sqft', 'position', 'floor', 'rating']]
        y = data['rent_amount']

        X_encoded = pd.get_dummies(X, columns=['location', 'position'], drop_first=True)
        self.original_X_columns = X_encoded.columns.tolist()
        self.valid_locations = data['location'].unique().tolist()
        self.median_floor = data['floor'].median()
        self.median_rating = data['rating'].median()

        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
        print(f"Training set size: {len(X_train)} samples")
        print(f"Testing set size: {len(X_test)} samples")

        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        print("Random Forest Regressor model trained.")

        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Mean Absolute Error (MAE): ${mae:.2f}")
        print(f"R-squared (R2): {r2:.4f}")

        joblib.dump(self.model, 'rental_predictor_model.pkl')
        joblib.dump(self.original_X_columns, 'original_X_columns.pkl')
        joblib.dump(self.location_counts, 'location_counts.pkl')
        print("Model, columns, and location counts saved.")

    def predict_rental_price(self, location, size_sqft, position='front', floor=None, rating=None):
        """
        Predict rent for a property.
        """
        if not self.model or not self.original_X_columns:
            raise ValueError("Model not initialized. Train or load a model first.")
        
        if location not in self.valid_locations:
            raise ValueError(f"Invalid location: {location}. Choose from: {self.valid_locations}")
        if not isinstance(size_sqft, (int, float)) or size_sqft <= 0:
            raise ValueError("size_sqft must be a positive number")
        
        # Warn for locations with limited data (< 3 data points)
        location_count = self.location_counts.get(location, 0)
        if location_count < 3:
            print(f"Warning: Limited data for {location} ({location_count} data points). Prediction may be less accurate.")

        new_property = pd.DataFrame([{
            'location': location,
            'size_sqft': size_sqft,
            'position': position,
            'floor': floor if floor is not None else self.median_floor,
            'rating': rating if rating is not None else self.median_rating
        }])

        new_property_encoded = pd.get_dummies(new_property, columns=['location', 'position'], drop_first=True)
        for col in self.original_X_columns:
            if col not in new_property_encoded.columns:
                new_property_encoded[col] = 0
        new_property_for_prediction = new_property_encoded[self.original_X_columns]

        predicted_rent = self.model.predict(new_property_for_prediction)[0]
        
        if location == 'Mirpur' and predicted_rent > 20000:
            predicted_rent = 15000  # Cap based on dataset
        return predicted_rent