import React, { useRef, useEffect, useState } from "react";
import * as d3 from "d3";

const DynamicGraph = ({ predictions }) => {
  const svgRef = useRef();
  const [chartData, setChartData] = useState(predictions);

  useEffect(() => {
    // Set up the SVG canvas dimensions
    const width = 600;
    const height = 400;
    const margin = { top: 20, right: 30, bottom: 30, left: 40 };

    // Clear previous SVG content
    d3.select(svgRef.current).selectAll("*").remove();

    // Create the SVG canvas
    const svg = d3
      .select(svgRef.current)
      .attr("width", width)
      .attr("height", height);

    // Set up x and y scales
    const xScale = d3
      .scaleLinear()
      .domain([0, chartData.length - 1])
      .range([margin.left, width - margin.right]);

    const yScale = d3
      .scaleLinear()
      .domain([0, d3.max(chartData, (d) => d.value)])
      .nice()
      .range([height - margin.bottom, margin.top]);

    // Add x and y axes
    svg
      .append("g")
      .attr("transform", `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(xScale).ticks(chartData.length));

    svg
      .append("g")
      .attr("transform", `translate(${margin.left},0)`)
      .call(d3.axisLeft(yScale));

    // Create a line generator
    const line = d3
      .line()
      .x((d, i) => xScale(i))
      .y((d) => yScale(d.value))
      .curve(d3.curveMonotoneX);

    // Add the line path
    svg
      .append("path")
      .datum(chartData)
      .attr("fill", "none")
      .attr("stroke", "steelblue")
      .attr("stroke-width", 2)
      .attr("d", line);
  }, [chartData]);

  // Update chart data when predictions prop changes
  useEffect(() => {
    setChartData(predictions);
  }, [predictions]);

  return <svg ref={svgRef}></svg>;
};

export default DynamicGraph;
