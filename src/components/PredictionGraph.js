import React, { useEffect, useState } from "react";
import axios from "axios";

import { StyledListElement } from "../styles/AppStyles";
import Typography from "@mui/material/Typography";

export default function PredictionGraph({aqi, category, advice}) {
  const [graphUrl, setGraphUrl] = useState(null);
  const [predictions, setPredictions] = useState([]);

  // Fetch predictions data
  /*const fetchPredictions = async () => {
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

  useEffect(() => {
    fetchPredictions();
    fetchGraph();
  }, []);
  };*/
  
  var goodElement = document.getElementById('good');
  var moderateElement = document.getElementById('moderate');
  var satisfactoryElement = document.getElementById('satisfactory');
  var poorElement = document.getElementById('poor');
  var veryPoorElement = document.getElementById('very-poor');
  var severeElement = document.getElementById('severe');
  
  if (goodElement != null)
  {
  goodElement.style.background = 'white';
  moderateElement.style.background = 'white';
  satisfactoryElement.style.background = 'white';
  poorElement.style.background = 'white';
  veryPoorElement.style.background = 'white';
  severeElement.style.background = 'white';
  }
  
  switch(category)
  {
	case 'good':
		goodElement.style.background = 'purple';
	break;
	
	case 'moderate':
		moderateElement.style.background = 'purple';
	break;
	
	case 'satisfactory':
		satisfactoryElement.style.background = 'purple';
	break;
	
	case 'poor':
		poorElement.style.background = 'purple';
	break;
	
	case 'very-poor':
		veryPoorElement.style.background = 'purple';
	break;
	
	case 'servere':
		severeElement.style.background = 'purple';
	break;
  }
    return (
    <div>
		<Typography>AQI: {aqi}</Typography>
		<Typography variant="h5">Category</Typography>
		<StyledListElement id="good"><Typography>Good</Typography></StyledListElement>
		<StyledListElement id="moderate"><Typography>Moderate</Typography></StyledListElement>
		<StyledListElement id="satisfactory"><Typography>Satisfactory</Typography></StyledListElement>
		<StyledListElement id="poor"><Typography>Poor</Typography></StyledListElement>
		<StyledListElement id="very-poor"><Typography>Very Poor</Typography></StyledListElement>
		<StyledListElement id="severe"><Typography>Severe</Typography></StyledListElement>
		<Typography>Health Advice: {advice}</Typography>
	</div>
  );
};

//export default PredictionGraph;

// good,moderate,satisfactory,poor,very poor,severe