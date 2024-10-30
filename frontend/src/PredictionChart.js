import React, { useState } from "react";

function AirQualityForm() {
  const [PM25, setPM25] = useState(50);
  const [PM10, setPM10] = useState(50);
  const [co, setCo] = useState(50);
  const [SO2, setSO2] = useState(50);
  const [NO2, setNO2] = useState(50);
  const [O3, setO3] = useState(50);
  const [modelChoice, setModelChoice] = useState("Classification");
  const [predictionResult, setPredictionResult] = useState(null); // State for prediction result

  const handleSubmit = async (event) => {
    event.preventDefault();
    const inputData = {
      PM25,
      PM10,
      co,
      SO2,
      NO2,
      O3,
    };

    try {
      const response = await fetch(
        `http://127.0.0.1:8000/predict?model_choice=${modelChoice}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(inputData),
        }
      );

      if (!response.ok) {
        throw new Error("Failed to fetch prediction");
      }

      const data = await response.json();
      setPredictionResult(data); // Store the prediction result in state
    } catch (error) {
      console.error("Error:", error);
      setPredictionResult({
        error: "Failed to get prediction. Please try again.",
      });
    }
  };

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <label>
          PM2.5:
          <input
            type="range"
            min="1"
            max="100"
            value={PM25}
            onChange={(e) => setPM25(Number(e.target.value))}
          />
          {PM25}
        </label>
        <br />

        <label>
          PM10:
          <input
            type="range"
            min="1"
            max="100"
            value={PM10}
            onChange={(e) => setPM10(Number(e.target.value))}
          />
          {PM10}
        </label>
        <br />

        <label>
          CO:
          <input
            type="range"
            min="1"
            max="100"
            value={co}
            onChange={(e) => setCo(Number(e.target.value))}
          />
          {co}
        </label>
        <br />

        <label>
          SO2:
          <input
            type="range"
            min="1"
            max="100"
            value={SO2}
            onChange={(e) => setSO2(Number(e.target.value))}
          />
          {SO2}
        </label>
        <br />

        <label>
          NO2:
          <input
            type="range"
            min="1"
            max="100"
            value={NO2}
            onChange={(e) => setNO2(Number(e.target.value))}
          />
          {NO2}
        </label>
        <br />

        <label>
          O3:
          <input
            type="range"
            min="1"
            max="100"
            value={O3}
            onChange={(e) => setO3(Number(e.target.value))}
          />
          {O3}
        </label>
        <br />

        <label>
          Model Choice:
          <select
            value={modelChoice}
            onChange={(e) => setModelChoice(e.target.value)}
          >
            <option value="Classification">Classification</option>
            <option value="Regression">Regression</option>
          </select>
        </label>
        <br />

        <button type="submit">Predict</button>
      </form>

      {/* Display Prediction Result */}
      {predictionResult && (
        <div>
          <h3>Prediction Result:</h3>
          {predictionResult.error ? (
            <p>{predictionResult.error}</p>
          ) : (
            <>
              <p>Rating: {predictionResult.rating}</p>
              <p>Rating Label: {predictionResult.rating_label}</p>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default AirQualityForm;
