import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { StyledPaper } from '../styles/AppStyles';
import Typography from '@mui/material/Typography';

function DataVisualization({ aqiCategory, inputData }) {
  const svgRefRadial = useRef();
  const svgRefInputs = useRef();

  // Updated AQI Categories with appropriate colors
  const aqiCategories = {
    "Good": 50,
    "Moderate": 100,
    "Satisfactory": 200,
    "Poor": 300,
    "Very Poor": 400,
    "Severe": 500,
  };

  const categoryColors = {
    "Good": "#4caf50",
    "Moderate": "#ffeb3b",
    "Satisfactory": "#ffa726",
    "Poor": "#ff5722",
    "Very Poor": "#d32f2f",
    "Severe": "#b71c1c"
  };

  useEffect(() => {
    // Radial Chart for AQI Category
    const svg = d3.select(svgRefRadial.current);
    const width = 300;
    const height = 300;
    const radius = Math.min(width, height) / 2;

    svg.attr("width", width).attr("height", height);
    svg.selectAll("*").remove();  // Clear previous render

    const pie = d3.pie().value(d => d.value);
    const data = Object.entries(aqiCategories).map(([key, value]) => ({
      category: key,
      value,
    }));

    const arc = d3.arc()
      .innerRadius(60)  // Makes it a donut chart
      .outerRadius(radius);

    const colorScale = d3.scaleOrdinal()
      .domain(Object.keys(aqiCategories))
      .range(Object.values(categoryColors));

    const arcs = svg.append("g")
      .attr("transform", `translate(${width / 2}, ${height / 2})`)
      .selectAll("arc")
      .data(pie(data))
      .enter()
      .append("path")
      .attr("d", arc)
      .attr("fill", d => colorScale(d.data.category))
      .attr("opacity", d => (d.data.category === aqiCategory ? 1 : 0.3));

    // Add text label for the selected AQI Category
    svg.append("text")
      .attr("x", width / 2)
      .attr("y", height / 2)
      .attr("text-anchor", "middle")
      .attr("font-size", "18px")
      .attr("fill", categoryColors[aqiCategory] || "#000")
      .text(aqiCategory || "Select a Category");
  }, [aqiCategory]);

  useEffect(() => {
    // Input Values Bar Chart with Enhanced Axes Styling
    const svgInputs = d3.select(svgRefInputs.current);
    const width = 600;
    const height = 300;
    const margin = { top: 20, right: 30, bottom: 50, left: 50 };  // Increased bottom margin

    svgInputs.attr('width', width).attr('height', height);
    svgInputs.selectAll("*").remove();  // Clear previous render

    const inputDataArray = Object.entries(inputData || {}).map(([key, value]) => ({
      parameter: key,
      value: value,
    }));

    const xScale = d3.scaleBand()
      .domain(inputDataArray.map(d => d.parameter))
      .range([margin.left, width - margin.right])
      .padding(0.3);

    const yScale = d3.scaleLinear()
      .domain([0, d3.max(inputDataArray, d => d.value) || 1])
      .range([height - margin.bottom, margin.top]);

    svgInputs.selectAll(".bar")
      .data(inputDataArray)
      .join("rect")
      .attr("class", "bar")
      .attr("x", d => xScale(d.parameter))
      .attr("y", d => yScale(d.value))
      .attr("width", xScale.bandwidth())
      .attr("height", d => height - margin.bottom - yScale(d.value))
      .attr("fill", "#3f51b5")
      .attr("rx", 5);

    svgInputs.selectAll(".bar-label")
      .data(inputDataArray)
      .join("text")
      .attr("class", "bar-label")
      .attr("x", d => xScale(d.parameter) + xScale.bandwidth() / 2)
      .attr("y", d => yScale(d.value) - 5)
      .attr("text-anchor", "middle")
      .attr("fill", "#333333")  // Dark color for contrast
      .attr("font-weight", "bold")
      .attr("font-size", "14px")  // Slightly larger font size
      .text(d => d.value);

    // Enhanced Axes Styling
    svgInputs.selectAll(".x-axis").remove();
    svgInputs.append("g")
      .attr("class", "x-axis")
      .attr("transform", `translate(0,${height - margin.bottom + 10})`)  // Adjusted for added margin
      .call(d3.axisBottom(xScale))
      .selectAll("text")
      .attr("transform", "rotate(-45)")  // Rotate labels for readability
      .style("text-anchor", "end")
      .style("font-size", "12px")
      .style("fill", "#333");

    svgInputs.selectAll(".y-axis").remove();
    svgInputs.append("g")
      .attr("class", "y-axis")
      .attr("transform", `translate(${margin.left},0)`)
      .call(d3.axisLeft(yScale).ticks(5))
      .selectAll("text")
      .style("font-size", "12px")
      .style("fill", "#333");

    svgInputs.selectAll(".tick line").style("stroke", "#ccc");  // Light gray gridlines
  }, [inputData]);

  return (
    <StyledPaper>
      <Typography variant="h6" gutterBottom>
        AQI Classification
      </Typography>
      <svg ref={svgRefRadial}></svg>

      <Typography variant="h6" gutterBottom style={{ marginTop: '20px' }}>
        Input Parameter Values
      </Typography>
      <svg ref={svgRefInputs}></svg>
    </StyledPaper>
  );
}

export default DataVisualization;
