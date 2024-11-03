import React, { useState } from "react";
import axios from "axios";
import InputForm from "./components/InputForm";
import DataVisualization from "./components/DataVisualization";
import DynamicGraph from "./components/DynamicGraph"; // Import DynamicGraph component
import PredictionGraph from "./components/PredictionGraph";
import { AppContainer, StyledPaper } from "./styles/AppStyles";
import Typography from "@mui/material/Typography";
import Grid from "@mui/material/Grid";
import Box from "@mui/material/Box";

function App() {
  const [aqiPrediction, setAqiPrediction] = useState(null);
  const [aqiCategory, setAqiCategory] = useState(null);
  const [healthAdvice, setHealthAdvice] = useState(null);
  const [inputData, setInputData] = useState({});
  const [predictions, setPredictions] = useState([]);
  const [error, setError] = useState(null);

  const handleFormSubmit = async (inputData) => {
    setInputData(inputData);
    setError(null);
    console.log("Sending input data:", inputData);
    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/predict",
        inputData,
        {
          headers: { "Content-Type": "application/json" },
        }
      );
      setAqiPrediction(response.data.rating);
      setAqiCategory(response.data.rating_label);
      setHealthAdvice(response.data.general_health_advice);
    } catch (error) {
      console.error("Error fetching prediction or classification:", error);
      setError("Failed to fetch data from backend. Please try again.");
      setAqiPrediction(null);
      setAqiCategory(null);
      setHealthAdvice(null);
    }
  };

  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        minHeight: "100vh",
        overflowY: "auto",
        padding: 2,
      }}
    >
      <Typography variant="h4" gutterBottom align="center">
        AQI Prediction and Classification
      </Typography>

      <Grid container spacing={2} justifyContent="center">
        {/* Input Form Section */}
        <Grid item xs={12} md={6}>
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
              <DataVisualization inputData={inputData} />
            </>
          )}
        </Grid>

        {/* Prediction Graph Section */}
        <Grid item xs={12} md={6}>
          <StyledPaper>
            <Typography variant="h4" gutterBottom align="center">
              Prediction Results
            </Typography>
            <PredictionGraph
              aqi={aqiPrediction}
              category={aqiCategory}
              advice={healthAdvice}
            />

            {/* Add DynamicGraph below PredictionGraph */}
            <Typography variant="h6" align="center" gutterBottom>
              All Predictions Over Time
            </Typography>
            <DynamicGraph />
          </StyledPaper>
        </Grid>
      </Grid>
    </Box>
  );
}

export default App;
