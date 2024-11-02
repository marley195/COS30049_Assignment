import React, { useEffect, useState } from "react";
import axios from "axios";

const PredictionGraph = () => {
  const [graphUrl, setGraphUrl] = useState(null);
  const [predictions, setPredictions] = useState([]);

  // Fetch predictions data
  const fetchPredictions = async () => {
    try {
      const response = await axios.get(
        "http://127.0.0.1:8000/predictions/plot",
        {
          responseType: "blob",
        }
      );
      setPredictions(response.data.predictions);
    } catch (error) {
      console.error("Error fetching predictions:", error);
    }
  };

  // Fetch the graph image URL
  const fetchGraph = async () => {
    try {
      const response = await axios.get(
        "http://127.0.0.1:8000/predictions/plot",
        {
          responseType: "blob", // Important to receive image data
        }
      );
      // Create a URL for the image blob
      const imageUrl = URL.createObjectURL(response.data);
      setGraphUrl(imageUrl);
    } catch (error) {
      console.error("Error fetching graph:", error);
    }
  };

  useEffect(() => {
    fetchPredictions();
    fetchGraph();
  }, []);

  return (
    <div>
      <h1>Prediction Graph</h1>
      {graphUrl && (
        <div>
          <img src={graphUrl} alt="Prediction Graph" />
        </div>
      )}
      <h2>Prediction List</h2>
      <ul>
        {predictions.map((prediction, index) => (
          <li key={index}>{prediction}</li>
        ))}
      </ul>
    </div>
  );
};

export default PredictionGraph;
