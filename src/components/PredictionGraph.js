import React, { useEffect, useState } from "react";
import axios from "axios";
import { StyledListElement } from "../styles/AppStyles";
import Typography from "@mui/material/Typography";

export default function PredictionGraph({ aqi, category, advice }) {
  const [graphUrl, setGraphUrl] = useState(null);
  const [predictions, setPredictions] = useState([]);

  // Uncomment and use these functions when you need to fetch predictions and graph data
  /*const fetchPredictions = async () => {
    try {
      const response = await axios.get("http://127.0.0.1:8000/predictions/plot", {
        responseType: "blob",
      });
      setPredictions(response.data.predictions);
    } catch (error) {
      console.error("Error fetching predictions:", error);
    }
  };

  const fetchGraph = async () => {
    try {
      const response = await axios.get("http://127.0.0.1:8000/predictions/plot", {
        responseType: "blob",
      });
      const imageUrl = URL.createObjectURL(response.data);
      setGraphUrl(imageUrl);
    } catch (error) {
      console.error("Error fetching graph:", error);
    }
  };

  useEffect(() => {
    fetchPredictions();
    fetchGraph();
  }, []);*/

  useEffect(() => {
    const categoryElements = {
      Good: document.getElementById("Good"),
      Moderate: document.getElementById("Moderate"),
      Unhealthy: document.getElementById("Unhealthy"),
      "Very Unhealthy": document.getElementById("Very Unhealthy"),
      Hazardous: document.getElementById("Hazardous"),
      Severe: document.getElementById("Severe"),
    };

    // Reset all backgrounds to white
    Object.values(categoryElements).forEach((element) => {
      if (element) element.style.background = "white";
    });

    // Set background color for the selected category
    if (categoryElements[category]) {
      categoryElements[category].style.background = "purple";
    }
  }, [category]);

  return (
    <div>
      <Typography>AQI: {aqi}</Typography>
      <Typography variant="h5">Category</Typography>
      <StyledListElement id="Good">
        <Typography>Good</Typography>
      </StyledListElement>
      <StyledListElement id="Moderate">
        <Typography>Moderate</Typography>
      </StyledListElement>
      <StyledListElement id="Unhealthy">
        <Typography>Unhealthy</Typography>
      </StyledListElement>
      <StyledListElement id="Very Unhealthy">
        <Typography>Very Unhealthy</Typography>
      </StyledListElement>
      <StyledListElement id="Hazardous">
        <Typography>Hazardous</Typography>
      </StyledListElement>
      <StyledListElement id="Severe">
        <Typography>Severe</Typography>
      </StyledListElement>
      <Typography>Health Advice: {advice}</Typography>
    </div>
  );
}
