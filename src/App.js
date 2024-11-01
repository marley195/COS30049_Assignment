import React, { useState } from "react";
import axios from "axios";
import InputForm from "./components/InputForm";
import DataVisualization from "./components/DataVisualization";
import { AppContainer, StyledPaper } from "./styles/AppStyles";
import Typography from "@mui/material/Typography";

function App() {
  const [aqiPrediction, setAqiPrediction] = useState(0);
  const [aqiCategory, setAqiCategory] = useState(0);
  const [inputData, setInputData] = useState({});
  const [error, setError] = useState(null); // Initialize as null to avoid confusio
  const handleFormSubmit = async (inputData) => {
    // Store input data immediately for visualization
    setInputData(inputData);
    setError(null); // Reset error message
    console.log("Sending input data:", inputData);
    try {
      // Send request to backend
      const response = await axios.post(
        "http://127.0.0.1:8000/predict", // Corrected model_choice
        inputData, // Send inputData directly
        {
          headers: { "Content-Type": "application/json" },
        }
      );
      console.log("Sending input data:", inputData);
      // Set response data in state
      setAqiPrediction(response.data[0]);
      console.log(response.data[0]);
      setAqiCategory(response.data[1]);
      console.log(response.data[1]);
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
        <InputForm onSubmit={handleFormSubmit} />
      </StyledPaper>

      {error && (
        <Typography variant="body1" color="error" gutterBottom>
          {error}
        </Typography>
      )}

      {Object.keys(inputData).length > 0 && (
        <>
          {aqiPrediction !== null && aqiPrediction !== undefined && (
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
