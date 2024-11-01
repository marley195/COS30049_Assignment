import React, { useState } from "react";
import axios from "axios";
import InputForm from "./components/InputForm";
import DataVisualization from "./components/DataVisualization";
import { AppContainer, StyledPaper } from "./styles/AppStyles";
import Typography from "@mui/material/Typography";

function App() {
  const [aqiPrediction, setAqiPrediction] = useState(null);
  const [aqiCategory, setAqiCategory] = useState(null);
  const [inputData, setInputData] = useState({});
  const [error, setError] = useState(null);

  const handleFormSubmit = async (inputData) => {
    setInputData(inputData);
    setError(null);

    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/predict_aqi",
        inputData
      );
      setAqiPrediction(response.data.aqi_prediction);
      setAqiCategory(response.data.aqi_category);
    } catch (error) {
      console.error("Error fetching prediction or classification:", error);
      setError("Failed to fetch data from backend. Please try again.");
      setAqiPrediction(null);
      setAqiCategory(null);
    }
  };

  return (
    <AppContainer>
      <Typography variant="h4" gutterBottom>
        AQI Prediction and Classification
      </Typography>
      <StyledPaper>
        <InputForm handleFormSubmit={handleFormSubmit} />
      </StyledPaper>

      {error && (
        <Typography variant="body1" color="error" gutterBottom>
          {error}
        </Typography>
      )}

      {/* Conditional rendering of the result section */}
      {Object.keys(inputData).length > 0 && (
        <>
          {aqiPrediction !== null && (
            <Typography variant="h6">
              Predicted AQI: {aqiPrediction.toFixed(2)}
            </Typography>
          )}
          {aqiCategory && (
            <Typography variant="h6">
              Air Quality Category: {aqiCategory}
            </Typography>
          )}
          <DataVisualization aqiCategory={aqiCategory} inputData={inputData} />
        </>
      )}
    </AppContainer>
  );
}

export default App;
