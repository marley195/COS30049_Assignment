import React, { useEffect, useState } from "react";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import axios from "axios";

// Register components needed for chart.js
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

export default function DynamicGraph() {
  const [chartData, setChartData] = useState(null);

  // Fetch predictions, this will also update the chart
  const fetchPredictions = async () => {
    try {
      const response = await axios.get("http://127.0.0.1:8000/predictions");
      const predictions = response.data.predictions;

      // Prepare data for the chart, this will include labels the new health advice and catggory to give further insight.
      const labels = predictions.map((_, index) => `Prediction ${index + 1}`);
      const aqiValues = predictions.map((entry) => entry.prediction);
      const categories = predictions.map((entry) => entry["Rating Bracket"]);
      const healthAdvice = predictions.map(
        (entry) => entry["General Health Advice"]
      );

      // Update the chart data, this will trigger a re-render and allows the graph to dynmaically update with new data.
      setChartData({
        labels,
        datasets: [
          {
            label: "Predicted AQI",
            data: aqiValues,
            borderColor: "rgba(75,192,192,1)",
            fill: false,
          },
        ],
        // Add tooltip data for each point
        tooltipData: predictions.map((entry, index) => ({
          category: categories[index],
          advice: healthAdvice[index],
        })),
      });
      // Catch error and print reason
    } catch (error) {
      console.error("Error fetching predictions:", error);
    }
  };

  // Allows the component to fect data every 5 seconds.
  useEffect(() => {
    fetchPredictions(); // Initial fetch

    const intervalId = setInterval(fetchPredictions, 5000); // Fetch data every 5 seconds

    // Clean up interval on component unmount
    return () => clearInterval(intervalId);
  }, []); // Empty dependency array ensures this runs only once
  // Return a loading message if chart data is not available
  if (!chartData) return <div>Loading chart...</div>;
  // Return the Line chart with the data that will show when the users hover over the chart plots.
  return (
    <div>
      <Line
        data={chartData}
        options={{
          plugins: {
            tooltip: {
              callbacks: {
                label: function (context) {
                  const index = context.dataIndex;
                  const aqi = context.raw;
                  const category = chartData.tooltipData[index].category;
                  const advice = chartData.tooltipData[index].advice;
                  return [
                    `AQI: ${aqi}`,
                    `Category: ${category}`,
                    `Health Advice: ${advice}`,
                  ];
                },
              },
            },
          },
        }}
      />
    </div>
  );
}
