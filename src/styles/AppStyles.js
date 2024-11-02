import { styled } from "@mui/material/styles";
import Container from "@mui/material/Container";
import Paper from "@mui/material/Paper";

export const AppContainer = styled(Container)(({ theme }) => ({
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  minHeight: "100vh",
  background: "linear-gradient(135deg, #2193b0, #6dd5ed)", // Gradient background
  color: "#ffffff", // White text color
  padding: "1rem",
  overflowY: "auto", // Enable scrolling for smaller screens

  [theme.breakpoints.down("sm")]: {
    padding: "0.5rem",
    minHeight: "auto",
    height: "100vh",
  },
}));

export const StyledPaper = styled(Paper)(({ theme }) => ({
  padding: "2rem",
  width: "100%",
  maxWidth: "800px",
  margin: "2rem 0",
  textAlign: "center",
  backgroundColor: "rgba(255, 255, 255, 0.9)", // Slightly transparent white background
  boxShadow: "0px 4px 12px rgba(0, 0, 0, 0.1)",
  borderRadius: "8px", // Rounded corners

  [theme.breakpoints.down("sm")]: {
    padding: "1rem",
    margin: "1rem 0",
  },
}));
