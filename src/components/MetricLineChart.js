import React, { useState, useEffect } from "react";
import * as d3 from "d3";

const MetricChart = ({ data }) => {
  const [levels, setLevels] = useState({});

  // Function to fetch levels from FastAPI
  const fetchLevels = async () => {
    const response = await fetch("http://127.0.0.1:8000/analyze_metrics", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    });
    const result = await response.json();
    setLevels(result.levels);
  };

  useEffect(() => {
    fetchLevels();
  }, [data]);

  useEffect(() => {
    // Create the chart with D3
    const svg = d3.select("#chart").attr("width", 600).attr("height", 400);

    const xScale = d3
      .scaleBand()
      .domain(Object.keys(data))
      .range([0, 600])
      .padding(0.4);

    const yScale = d3
      .scaleLinear()
      .domain([0, 120]) // Adjust based on max value
      .range([400, 0]);

    svg
      .selectAll(".bar")
      .data(Object.entries(data))
      .enter()
      .append("rect")
      .attr("class", "bar")
      .attr("x", ([metric]) => xScale(metric))
      .attr("y", ([, value]) => yScale(value))
      .attr("width", xScale.bandwidth())
      .attr("height", ([, value]) => 400 - yScale(value))
      .attr("fill", "steelblue")
      .on("mouseover", (event, [metric]) => {
        // Show level tooltip based on FastAPI response
        const level = levels[metric] || "Unknown";
        d3.select("#tooltip")
          .style("left", `${event.pageX + 10}px`)
          .style("top", `${event.pageY - 10}px`)
          .style("display", "inline-block")
          .html(
            `<strong>${metric}</strong><br>Value: ${data[metric]}<br>Level: ${level}`
          );
      })
      .on("mouseout", () => {
        d3.select("#tooltip").style("display", "none");
      });
  }, [levels, data]);

  return (
    <div>
      <svg id="chart"></svg>
      <div
        id="tooltip"
        style={{
          position: "absolute",
          display: "none",
          background: "white",
          border: "1px solid #ccc",
          padding: "10px",
          borderRadius: "5px",
        }}
      ></div>
    </div>
  );
};

export default MetricChart;
