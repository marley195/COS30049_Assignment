import React, { useState } from "react";
import { Slider, Button, Typography, Box } from "@mui/material";

function InputForm({ onSubmit }) {
  // State for each pollutant
  const [benzene, setBenzene] = useState(0);
  const [co, setCo] = useState(0);
  const [nh3, setNh3] = useState(0);
  const [no, setNo] = useState(0);
  const [no2, setNo2] = useState(0);
  const [nox, setNox] = useState(0);
  const [o3, setO3] = useState(0);
  const [pm10, setPm10] = useState(0);
  const [pm2_5, setPm2_5] = useState(0);
  const [so2, setSo2] = useState(0);
  const [toluene, setToluene] = useState(0);
  const [xylene, setXylene] = useState(0);

  // Color ranges for each AQI classification level
  const getColor = (value) => {
    if (value <= 50) return "#4caf50"; // Good
    if (value <= 100) return "#ffeb3b"; // Moderate
    if (value <= 200) return "#ffa726"; // Satisfactory
    if (value <= 300) return "#ff5722"; // Poor
    if (value <= 400) return "#d32f2f"; // Very Poor
    return "#b71c1c"; // Severe
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit({
      Benzene: benzene,
      CO: co,
      NH3: nh3,
      NO: no,
      NO2: no2,
      NOx: nox,
      O3: o3,
      PM10: pm10,
      "PM2.5": pm2_5, // Note underscore for consistency with API model
      SO2: so2,
      Toluene: toluene,
      Xylene: xylene,
    });
  };

  return (
    <form onSubmit={handleSubmit}>
      <Box sx={{ mb: 2 }}>
        <Typography gutterBottom>Benzene (µg/m³)</Typography>
        <Slider
          value={benzene}
          onChange={(e, val) => setBenzene(val)}
          min={0}
          max={50}
          sx={{ color: getColor(benzene) }}
        />
        <Typography>Current Value: {benzene} µg/m³</Typography>
      </Box>

      <Box sx={{ mb: 2 }}>
        <Typography gutterBottom>CO (mg/m³)</Typography>
        <Slider
          value={co}
          onChange={(e, val) => setCo(val)}
          min={0}
          max={50}
          step={0.1}
          sx={{ color: getColor(co * 10) }} // Scaling to match AQI colors
        />
        <Typography>Current Value: {co} mg/m³</Typography>
      </Box>

      <Box sx={{ mb: 2 }}>
        <Typography gutterBottom>NH3 (µg/m³)</Typography>
        <Slider
          value={nh3}
          onChange={(e, val) => setNh3(val)}
          min={0}
          max={400}
          sx={{ color: getColor(nh3) }}
        />
        <Typography>Current Value: {nh3} µg/m³</Typography>
      </Box>

      <Box sx={{ mb: 2 }}>
        <Typography gutterBottom>NO (µg/m³)</Typography>
        <Slider
          value={no}
          onChange={(e, val) => setNo(val)}
          min={0}
          max={300}
          sx={{ color: getColor(no) }}
        />
        <Typography>Current Value: {no} µg/m³</Typography>
      </Box>

      <Box sx={{ mb: 2 }}>
        <Typography gutterBottom>NO2 (µg/m³)</Typography>
        <Slider
          value={no2}
          onChange={(e, val) => setNo2(val)}
          min={0}
          max={200}
          sx={{ color: getColor(no2) }}
        />
        <Typography>Current Value: {no2} µg/m³</Typography>
      </Box>

      <Box sx={{ mb: 2 }}>
        <Typography gutterBottom>NOx (µg/m³)</Typography>
        <Slider
          value={nox}
          onChange={(e, val) => setNox(val)}
          min={0}
          max={300}
          sx={{ color: getColor(nox) }}
        />
        <Typography>Current Value: {nox} µg/m³</Typography>
      </Box>

      <Box sx={{ mb: 2 }}>
        <Typography gutterBottom>O3 (µg/m³)</Typography>
        <Slider
          value={o3}
          onChange={(e, val) => setO3(val)}
          min={0}
          max={200}
          sx={{ color: getColor(o3) }}
        />
        <Typography>Current Value: {o3} µg/m³</Typography>
      </Box>

      <Box sx={{ mb: 2 }}>
        <Typography gutterBottom>PM10 (µg/m³)</Typography>
        <Slider
          value={pm10}
          onChange={(e, val) => setPm10(val)}
          min={0}
          max={600}
          sx={{ color: getColor(pm10) }}
        />
        <Typography>Current Value: {pm10} µg/m³</Typography>
      </Box>

      <Box sx={{ mb: 2 }}>
        <Typography gutterBottom>PM2_5 (µg/m³)</Typography>
        <Slider
          value={pm2_5}
          onChange={(e, val) => setPm2_5(val)}
          min={0}
          max={500}
          sx={{ color: getColor(pm2_5) }}
        />
        <Typography>Current Value: {pm2_5} µg/m³</Typography>
      </Box>

      <Box sx={{ mb: 2 }}>
        <Typography gutterBottom>SO2 (µg/m³)</Typography>
        <Slider
          value={so2}
          onChange={(e, val) => setSo2(val)}
          min={0}
          max={300}
          sx={{ color: getColor(so2) }}
        />
        <Typography>Current Value: {so2} µg/m³</Typography>
      </Box>

      <Box sx={{ mb: 2 }}>
        <Typography gutterBottom>Toluene (µg/m³)</Typography>
        <Slider
          value={toluene}
          onChange={(e, val) => setToluene(val)}
          min={0}
          max={100}
          sx={{ color: getColor(toluene) }}
        />
        <Typography>Current Value: {toluene} µg/m³</Typography>
      </Box>

      <Box sx={{ mb: 2 }}>
        <Typography gutterBottom>Xylene (µg/m³)</Typography>
        <Slider
          value={xylene}
          onChange={(e, val) => setXylene(val)}
          min={0}
          max={100}
          sx={{ color: getColor(xylene) }}
        />
        <Typography>Current Value: {xylene} µg/m³</Typography>
      </Box>

      <Button
        variant="contained"
        color="primary"
        fullWidth
        type="submit"
        sx={{ mt: 2 }}
      >
        Predict AQI
      </Button>
    </form>
  );
}

export default InputForm;
